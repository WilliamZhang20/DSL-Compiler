#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <iostream>

using namespace llvm;
using namespace llvm::orc;

int main() {
    // Initialize native target
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

    // Thread-safe context must own the LLVMContext
    auto TSCtx = std::make_unique<LLVMContext>();
    LLVMContext &Context = *TSCtx;

    // Create module
    auto M = std::make_unique<Module>("blaze_session0", Context);
    IRBuilder<> Builder(Context);

    // Define function: i32 add(i32, i32)
    auto *Int32 = Type::getInt32Ty(Context);
    auto *FuncTy = FunctionType::get(Int32, {Int32, Int32}, false);
    auto *AddFunc = Function::Create(FuncTy, Function::ExternalLinkage, "add", M.get());

    // Name arguments
    auto ArgIt = AddFunc->arg_begin();
    Value *X = &*ArgIt++; X->setName("x");
    Value *Y = &*ArgIt++; Y->setName("y");

    // Entry block
    BasicBlock *BB = BasicBlock::Create(Context, "entry", AddFunc);
    Builder.SetInsertPoint(BB);
    Value *Sum = Builder.CreateAdd(X, Y, "sum");
    Builder.CreateRet(Sum);

    verifyFunction(*AddFunc);
    M->print(errs(), nullptr);

    // Create the JIT
    auto JIT = cantFail(LLJITBuilder().create());

    // Load module into JIT
    cantFail(JIT->addIRModule(
        ThreadSafeModule(std::move(M), ThreadSafeContext(std::move(TSCtx)))
    ));

    // Lookup symbol
    auto AddSym = cantFail(JIT->lookup("add"));

    // Convert to function pointer (LLVM 18)
    int (*AddPtr)(int, int) = AddSym.toPtr<int (*)(int, int)>();

    // Run
    std::cout << "\n--- Running JIT-compiled code ---\n";
    std::cout << "5 + 7  = " << AddPtr(5, 7)   << "\n";
    std::cout << "20 + 3 = " << AddPtr(20, 3)  << "\n";
    std::cout << "0 + 0  = " << AddPtr(0, 0)   << "\n";

    return 0;
}