//===-- MemorySlotInterfaces.td - MemorySlot interfaces ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_MEMORYSLOTINTERFACES
#define MLIR_INTERFACES_MEMORYSLOTINTERFACES

include "mlir/IR/OpBase.td"

def PromotableAllocationOpInterface
    : OpInterface<"PromotableAllocationOpInterface"> {
  let description = [{
    Describes an operation allocating a memory slot that can be promoted into
    SSA values.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns a list of memory slots for which promotion should be attempted.
        This only considers the local semantics of the allocator, ignoring
        whether the slot pointer is properly used or not. This allocator is the
        "owner" of the returned slots, meaning no two allocators should return
        the same slot. The content of the memory slot must only be reachable
        using loads and stores to the provided slot pointer, no aliasing is
        allowed.

        Promotion of the slot will lead to the slot pointer no longer being
        used, leaving the content of the memory slot unreachable.
      }], "::llvm::SmallVector<::mlir::MemorySlot>", "getPromotableSlots",
      (ins)
    >,
    InterfaceMethod<[{
        Provides the default Value of this memory slot. The provided Value
        will be used as the reaching definition of loads done before any store.
        This Value must outlive the promotion and dominate all the uses of this
        slot's pointer. The provided builder can be used to create the default
        value on the fly.

        The builder is located at the beginning of the block where the slot
        pointer is defined.
      }], "::mlir::Value", "getDefaultValue",
      (ins "const ::mlir::MemorySlot &":$slot, "::mlir::OpBuilder &":$builder)
    >,
    InterfaceMethod<[{
        Hook triggered for every new block argument added to a block.
        This will only be called for slots declared by this operation.

        The builder is located at the beginning of the block on call.
      }],
      "void", "handleBlockArgument",
      (ins
        "const ::mlir::MemorySlot &":$slot,
        "::mlir::BlockArgument":$argument,
        "::mlir::OpBuilder &":$builder
      )
    >,
    InterfaceMethod<[{
        Hook triggered once the promotion of a slot is complete. This can
        also clean up the created default value if necessary.
        This will only be called for slots declared by this operation.
      }],
      "void", "handlePromotionComplete",
      (ins "const ::mlir::MemorySlot &":$slot, "::mlir::Value":$defaultValue)
    >,
  ];
}

def PromotableMemOpInterface : OpInterface<"PromotableMemOpInterface"> {
  let description = [{
    Describes an operation that can load from memory slots and/or store
    to memory slots. Loads and stores must be of whole values of the same
    type as the slot itself.

    For a memory operation on a slot to be valid, it must operate on the slot
    pointer *only as a pointer to an element of the type of the slot*.

    If the same operation does both loads and stores on the same slot, the
    load must semantically happen first.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Gets whether this operation loads from the specified slot.
      }],
      "bool", "loadsFrom",
      (ins "const ::mlir::MemorySlot &":$slot)
    >,
    InterfaceMethod<[{
        Gets the value stored to the provided memory slot, or returns a null
        value if this operation does not store to this slot. An operation
        storing a value to a slot must always be able to provide the value it
        stores. This method is only called on operations that use the slot.
      }],
      "::mlir::Value", "getStored",
      (ins "const ::mlir::MemorySlot &":$slot)
    >,
    InterfaceMethod<[{
        Checks that this operation can be promoted to no longer use the provided
        blocking uses, in the context of promoting `slot`.

        If the removal procedure of the use will require that other uses get
        removed, that dependency should be added to the `newBlockingUses`
        argument. Dependent uses must only be uses of results of this operation.
      }], "bool", "canUsesBeRemoved",
      (ins "const ::mlir::MemorySlot &":$slot,
           "const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> &":$blockingUses,
           "::llvm::SmallVectorImpl<::mlir::OpOperand *> &":$newBlockingUses)
    >,
    InterfaceMethod<[{
        Transforms IR to ensure that the current operation does not use the
        provided memory slot anymore. `reachingDefinition` contains the value
        currently stored in the provided memory slot, immediately before the
        current operation.

        During the transformation, *no operation should be deleted*.
        The operation can only schedule its own deletion by returning the
        appropriate `DeletionKind`. The deletion must be legal assuming the
        blocking uses passed through the `newBlockingUses` list in
        `canUseBeRemoved` have been removed.

        After calling this method, the blocking uses should have disappeared
        or this operation should have scheduled its own deletion.

        This method will only be called after ensuring promotion is allowed via
        `canUseBeRemoved`. The requested blocking use removal may or may not
        have been done at the point of calling this method, but it will be done
        eventually.

        The builder is located after the promotable operation on call.
      }],
      "::mlir::DeletionKind",
      "removeBlockingUses",
      (ins "const ::mlir::MemorySlot &":$slot,
           "const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> &":$blockingUses,
           "::mlir::OpBuilder &":$builder,
           "::mlir::Value":$reachingDefinition)
    >,
  ];
}

def PromotableOpInterface : OpInterface<"PromotableOpInterface"> {
  let description = [{
    Describes an operation that can be transformed or deleted so it no longer
    uses a provided value (blocking use), in case this would allow the promotion
    of a memory slot.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Checks that this operation can be promoted to no longer use the provided
        blocking uses, in order to allow optimization.

        If the removal procedure of the use will require that other uses get
        removed, that dependency should be added to the `newBlockingUses`
        argument. Dependent uses must only be uses of results of this operation.
      }], "bool", "canUsesBeRemoved",
      (ins "const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> &":$blockingUses,
           "::llvm::SmallVectorImpl<::mlir::OpOperand *> &":$newBlockingUses)
    >,
    InterfaceMethod<[{
        Transforms IR to ensure that the current operation does not use the
        provided blocking uses anymore. In contrast to
        `PromotableMemOpInterface`, operations implementing this interface
        must not need access to the reaching definition of the content of the
        slot.

        During the transformation, *no operation should be deleted*.
        The operation can only schedule its own deletion by returning the
        appropriate `DeletionKind`. The deletion must be legal assuming the
        blocking uses passed through the `newBlockingUses` list in
        `canUseBeRemoved` have been removed.

        After calling this method, the blocking uses should have disappeared
        or this operation should have scheduled its own deletion.

        This method will only be called after ensuring promotion is allowed via
        `canUseBeRemoved`. The requested blocking use removal may or may not
        have been done at the point of calling this method, but it will be done
        eventually.

        The builder is located after the promotable operation on call.
      }],
      "::mlir::DeletionKind",
      "removeBlockingUses",
      (ins "const ::llvm::SmallPtrSetImpl<mlir::OpOperand *> &":$blockingUses,
           "::mlir::OpBuilder &":$builder)
    >,
  ];
}

def DestructibleAllocationOpInterface
  : OpInterface<"DestructibleAllocationOpInterface"> {
  let description = [{
    Describes operations allocating memory slots of aggregates that can be
    destructed into multiple smaller allocations.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns the list of slots for which destruction should be attempted,
        specifying in which way the slot should be destructed into subslots. The
        subslots are indexed by attributes. This computes the type of the
        pointers of each subslots to be generated. The type of the memory slot
        must implement `DestructibleTypeInterface`.
      }],
      "::llvm::SmallVector<::mlir::DestructibleMemorySlot>",
      "getDestructibleSlots",
      (ins)
    >,
    InterfaceMethod<[{
        Destructures this slot into multiple subslots. The newly generated slots
        may belong to a different allocator. The original slot must still exist
        at the end of this call.

        The builder is located at the beginning of the block where the slot
        pointer is defined.
      }],
      "::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot>",
      "destruct",
      (ins "const ::mlir::DestructibleMemorySlot &":$slot,
           "::llvm::SmallPtrSetImpl<::mlir::Attribute> &":$usedIndices,
           "::mlir::OpBuilder &":$builder)
    >,
    InterfaceMethod<[{
        Hook triggered once the destruction of a slot is complete, meaning the
        original slot is no longer being refered to and could be deleted.
        This will only be called for slots declared by this operation.
      }],
      "void", "handleDestructionComplete",
      (ins "const ::mlir::DestructibleMemorySlot &":$slot)
    >,
  ];
}

def TypeSafeOpInterface : OpInterface<"TypeSafeOpInterface"> {
  let description = [{
    Describes operations using memory slots in a type-safe manner.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns whether all accesses in this operation to the provided slot are
        done in a type-safe manner. To be type-safe, the access must only load
        the value in this type as the type of the slot, and without assuming any
        context around the slot. For example, a type-safe load must not load
        outside the bounds of the slot.

        If the type-safety of the accesses depends on the type-safety of the
        accesses to further memory slots, the result of this method will be
        conditioned to the type-safety of the accesses to the slots added by
        this method to `mustBeSafelyUsed`.
      }],
      "::mlir::LogicalResult",
      "ensureOnlyTypeSafeAccesses",
      (ins "const ::mlir::MemorySlot &":$slot,
           "::mlir::SmallVectorImpl<::mlir::MemorySlot> &":$mustBeSafelyUsed)
    >
  ];
}

def DestructibleAccessorOpInterface
  : OpInterface<"DestructibleAccessorOpInterface"> {
  let description = [{
    Describes operations that can access a sub-element of a destructible slot.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        For a given destructible memory slot, returns whether this operation can
        rewire its uses of the slot to use the slots generated after
        destruction. This may involve creating new operations, and usually
        amounts to checking the pointer types match.

        This method must also register the indices it will access within the
        `usedIndices` set. If the accessor generates new slots mapping to
        subelements, they must be registered in `mustBeSafelyUsed` to ensure
        they are used in a locally type-safe manner.
      }],
      "bool",
      "canRewire",
      (ins "const ::mlir::DestructibleMemorySlot &":$slot,
           "::llvm::SmallPtrSetImpl<::mlir::Attribute> &":$usedIndices,
           "::mlir::SmallVectorImpl<::mlir::MemorySlot> &":$mustBeSafelyUsed)
    >,
    InterfaceMethod<[{
        Rewires the use of a slot to the generated subslots, without deleting
        any operation. Returns whether the accessor should be deleted.
      }],
      "::mlir::DeletionKind",
      "rewire",
      (ins "const ::mlir::DestructibleMemorySlot &":$slot,
           "::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> &":$subslots)
    >
  ];
}

def DestructibleTypeInterface
  : TypeInterface<"DestructibleTypeInterface"> {
  let description = [{
    Describes a type that can be broken down into indexable sub-element types.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Destructs the type into subelements into a map of index attributes to
        types of subelements. Returns nothing if the type cannot be destructed.
      }],
      "::llvm::Optional<::llvm::DenseMap<::mlir::Attribute, ::mlir::Type>>",
      "getDestructedLayout",
      (ins)
    >,
    InterfaceMethod<[{
        Indicates which type is held at the provided index, returning a null
        Type if no type could be computed. While this can return information
        even when the type cannot be completely destructed, it must be coherent
        with the values returned by `destruct` when they exist.
      }],
      "::mlir::Type",
      "getTypeAtIndex",
      (ins "::mlir::Attribute":$index)
    >
  ];
}

#endif // MLIR_INTERFACES_MEMORYSLOTINTERFACES
