// RUN: heir-opt %s "--annotate-module=backend=openfhe scheme=bgv" "--mlir-to-bgv=ciphertext-degree=8192" | FileCheck %s

// CHECK: @func
func.func @func(%x: i16 {secret.secret}, %y: i16 {secret.secret}) -> (i16) {
    %0 = arith.addi %x, %y : i16
    %1 = arith.subi %x, %y : i16
    %2 = arith.muli %x, %y : i16
    %3 = arith.muli %0, %1 : i16
    %4 = arith.addi %3, %2 : i16
    func.return %4 : i16
}
