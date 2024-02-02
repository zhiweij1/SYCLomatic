//===--------------- BarrierFenceSpaceAnalyzer.h --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_BARRIER_FENCE_SPACE_ANALYZER_H
#define DPCT_BARRIER_FENCE_SPACE_ANALYZER_H

#include "Utility.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include <map>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

template <> struct std::hash<clang::SourceRange> {
  std::size_t operator()(const clang::SourceRange &SR) const noexcept {
    return llvm::hash_combine(SR.getBegin().getRawEncoding(),
                              SR.getEnd().getRawEncoding());
  }
};

namespace clang {
namespace dpct {

struct BarrierFenceSpaceAnalyzerResult {
  BarrierFenceSpaceAnalyzerResult() {}
  BarrierFenceSpaceAnalyzerResult(bool CanUseLocalBarrier,
                                  bool CanUseLocalBarrierWithCondition,
                                  bool MayDependOn1DKernel,
                                  std::string GlobalFunctionName,
                                  std::string Condition = "")
      : CanUseLocalBarrier(CanUseLocalBarrier),
        CanUseLocalBarrierWithCondition(CanUseLocalBarrierWithCondition),
        MayDependOn1DKernel(MayDependOn1DKernel),
        GlobalFunctionName(GlobalFunctionName), Condition(Condition) {}
  bool CanUseLocalBarrier = false;
  bool CanUseLocalBarrierWithCondition = false;
  bool MayDependOn1DKernel = false;
  std::string GlobalFunctionName;
  std::string Condition;
};

using Ranges = std::unordered_set<SourceRange>;
struct SyncCallInfo {
  SyncCallInfo() {}
  SyncCallInfo(Ranges Predecessors, Ranges Successors)
      : Predecessors(Predecessors), Successors(Successors){};
  Ranges Predecessors;
  Ranges Successors;
};

#define VISIT_NODE(CLASS)                                                      \
  bool Visit(const CLASS *Node);                                               \
  void PostVisit(const CLASS *FS);                                             \
  bool Traverse##CLASS(CLASS *Node) {                                          \
    if (!Visit(Node))                                                          \
      return false;                                                            \
    if (!RecursiveASTVisitor<BarrierFenceSpaceAnalyzer>::Traverse##CLASS(      \
            Node))                                                             \
      return false;                                                            \
    PostVisit(Node);                                                           \
    return true;                                                               \
  }

class BarrierFenceSpaceAnalyzer
    : public RecursiveASTVisitor<BarrierFenceSpaceAnalyzer> {
public:
  bool shouldVisitImplicitCode() const { return true; }
  bool shouldTraversePostOrder() const { return false; }

  VISIT_NODE(GotoStmt)
  VISIT_NODE(LabelStmt)
  VISIT_NODE(MemberExpr)
  VISIT_NODE(CXXDependentScopeMemberExpr)

  virtual BarrierFenceSpaceAnalyzerResult
  analyze(const CallExpr *CE, bool SkipCacheInAnalyzer = false) = 0;

protected:
  enum class AccessMode : int { Read = 0, Write, ReadWrite };
  struct DREInfo {
    DREInfo(const DeclRefExpr *DRE, SourceLocation SL, AccessMode AM)
        : DRE(DRE), SL(SL), AM(AM) {}
    const DeclRefExpr *DRE;
    SourceLocation SL;
    AccessMode AM;
    bool operator<(const DREInfo &Other) const { return DRE < Other.DRE; }
  };

  virtual std::pair<std::set<const DeclRefExpr *>, std::set<const VarDecl *>>
  isAssignedToAnotherDREOrVD(const DeclRefExpr *) = 0;
  virtual bool isAccessingMemory(const DeclRefExpr *) = 0;
  virtual AccessMode getAccessKind(const DeclRefExpr *) = 0;
  virtual std::tuple<bool /*CanUseLocalBarrier*/,
                     bool /*CanUseLocalBarrierWithCondition*/,
                     std::string /*Condition*/>
  isSafeToUseLocalBarrier(
      const std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap,
      const SyncCallInfo &SCI) = 0;
  virtual bool hasOverlappingAccessAmongWorkItems(int KernelDim,
                                                  const DeclRefExpr *DRE) = 0;
  virtual void constructDefUseMap() = 0;
  virtual void simplifyMap(
      std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap) = 0;
  virtual std::string isAnalyzableWriteInLoop(
      const std::set<const DeclRefExpr *> &WriteInLoopDRESet) = 0;

  bool containsMacro(const SourceLocation &SL, const SyncCallInfo &SCI);
  bool isInRanges(SourceLocation SL, Ranges Ranges);
  class TypeAnalyzer {
  public:
    enum class ParamterTypeKind : int {
      NeedAnalysis = 0,
      CanSkipAnalysis,
      Unsupported
    };
    ParamterTypeKind getInputParamterTypeKind(clang::QualType QT) {
      bool Res = canBeAnalyzed(QT.getTypePtr());
      if (!Res)
        return ParamterTypeKind::Unsupported;
      if (PointerLevel) {
        if (IsConstPtr)
          return ParamterTypeKind::CanSkipAnalysis;
        return ParamterTypeKind::NeedAnalysis;
      }
      return ParamterTypeKind::CanSkipAnalysis;
    }

  private:
    int PointerLevel = 0;
    bool IsConstPtr = false;
    bool IsClass = false;
    bool canBeAnalyzed(const clang::Type *TypePtr) {
      switch (TypePtr->getTypeClass()) {
      case clang::Type::TypeClass::ConstantArray:
        return canBeAnalyzed(dyn_cast<clang::ConstantArrayType>(TypePtr)
                                 ->getElementType()
                                 .getTypePtr());
      case clang::Type::TypeClass::Pointer:
        PointerLevel++;
        if (PointerLevel >= 2 || IsClass)
          return false;
        IsConstPtr = TypePtr->getPointeeType().isConstQualified();
        return canBeAnalyzed(TypePtr->getPointeeType().getTypePtr());
      case clang::Type::TypeClass::Elaborated:
        return canBeAnalyzed(
            dyn_cast<clang::ElaboratedType>(TypePtr)->desugar().getTypePtr());
      case clang::Type::TypeClass::Typedef:
        return canBeAnalyzed(dyn_cast<clang::TypedefType>(TypePtr)
                                 ->getDecl()
                                 ->getUnderlyingType()
                                 .getTypePtr());
      case clang::Type::TypeClass::Record:
        IsClass = true;
        if (PointerLevel &&
            isUserDefinedDecl(dyn_cast<clang::RecordType>(TypePtr)->getDecl()))
          return false;
        for (const auto &Field :
             dyn_cast<clang::RecordType>(TypePtr)->getDecl()->fields()) {
          if (!canBeAnalyzed(Field->getType().getTypePtr())) {
            return false;
          }
        }
        return true;
      case clang::Type::TypeClass::SubstTemplateTypeParm:
        return canBeAnalyzed(dyn_cast<clang::SubstTemplateTypeParmType>(TypePtr)
                                 ->getReplacementType()
                                 .getTypePtr());
      default:
        if (TypePtr->isFundamentalType())
          return true;
        else
          return false;
      }
    }
  };
};

class IntraproceduralAnalyzer : public BarrierFenceSpaceAnalyzer {};
class InterproceduralAnalyzer : public BarrierFenceSpaceAnalyzer {
public:
  BarrierFenceSpaceAnalyzerResult
  analyze(const CallExpr *CE, bool SkipCacheInAnalyzer = false) override;

  VISIT_NODE(ForStmt)
  VISIT_NODE(DoStmt)
  VISIT_NODE(WhileStmt)
  VISIT_NODE(CallExpr)
  VISIT_NODE(DeclRefExpr)
  VISIT_NODE(CXXConstructExpr)

private:
  std::pair<std::set<const DeclRefExpr *>, std::set<const VarDecl *>>
  isAssignedToAnotherDREOrVD(const DeclRefExpr *) override;
  bool isAccessingMemory(const DeclRefExpr *) override;
  AccessMode getAccessKind(const DeclRefExpr *) override;
  std::tuple<bool /*CanUseLocalBarrier*/,
             bool /*CanUseLocalBarrierWithCondition*/,
             std::string /*Condition*/>
  isSafeToUseLocalBarrier(
      const std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap,
      const SyncCallInfo &SCI) override;
  bool hasOverlappingAccessAmongWorkItems(int KernelDim,
                                          const DeclRefExpr *DRE) override;
  void constructDefUseMap() override;
  void simplifyMap(
      std::map<const ParmVarDecl *, std::set<DREInfo>> &DefDREInfoMap) override;
  std::string isAnalyzableWriteInLoop(
      const std::set<const DeclRefExpr *> &WriteInLoopDRESet) override;

  std::vector<std::pair<const CallExpr *, SyncCallInfo>> SyncCallsVec;
  std::deque<SourceRange> LoopRange;
  int KernelDim = 3;          // 3 or 1
  int KernelCallBlockDim = 3; // 3 or 1
  const FunctionDecl *FD = nullptr;
  std::string GlobalFunctionName;
  std::unordered_map<const ParmVarDecl *, std::set<const DeclRefExpr *>>
      DefUseMap;
  std::string CELoc;
  std::string FDLoc;
  /// (FD location, (Call location, result))
  static std::unordered_map<
      std::string,
      std::unordered_map<std::string, BarrierFenceSpaceAnalyzerResult>>
      CachedResults;
  bool SkipCacheInAnalyzer = false;
  bool MayDependOn1DKernel = false;
  std::set<const Expr *> DeviceFunctionCallArgs;
  bool IsDifferenceBetweenThreadIdxXAndIndexConstant = false;
  // This map contains pairs meet below pattern:
  // loop {
  //   ...
  //   DRE[idx] = ...;
  //   ...
  //   idx += step;
  //   ...
  // }
  std::map<const DeclRefExpr *, std::string> DREIncStepMap;
};
#undef VISIT_NODE

} // namespace dpct
} // namespace clang

#endif // DPCT_BARRIER_FENCE_SPACE_ANALYZER_H
