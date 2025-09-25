# ECMAScript atomics

This library provides unsafe (and potentially unsound) APIs for interacting
with memory that adheres to the
[ECMAScript memory model](https://tc39.es/ecma262/#sec-memory-model). The
important difference between the Rust/C++ and ECMAScript memory models is that
ECMAScript allows data races between atomic and non-atomic reads and writes, as
well as between mismatched reads and writes (ie. different sizes, atomic or
not). These sorts of data races are strictly undefined behaviour in Rust.

Because it is undefined behaviour to perform such actions in Rust, the
ECMAScript memory model interactions must happen such that from Rust's
perspective the memory participating in data races is not memory at all, and
the operations performing potentially racing reads and writes are outside of
the Rust compiler's view.

The first step to satisfying these requirements is to perform all the
operations either behind an FFI layer, or using inline assembly. This library's
choice is to use the latter option. The second step needed is to ensure that
from Rust's perspective, the memory participating in data races cannot be
considered as memory for the duration of the racy behaviour. This means
laundering the owning pointer of this memory through an inline assembly block
that is intended to signal to the Rust compiler that the memory may have been
deallocated, and then ensuring that the new pointer received from the
laundering is itself never used to perform memory reads or writes within Rust,
thus ensuring that the compiler cannot reason from example that the memory is
still allocated.

This leads to a concept of the three stages of ECMAScript memory, separated by
explicit fences injected by the `RacyMemory::enter` and `RacyMemory::exit`
method calls.

## First stage – initialisation

This library does not offer ways to allocate memory. All memory allocation
and initialisation must happen outside of this library (this happens to also
correspond with the ECMAScript memory model ordering of `INIT`). The likely
scenario is that such memory is initialised in Rust's view (if using eg. `Box`
or `alloc`) though it is possible to avoid that using lower level APIs (eg.
mmap can be used to allocate memory zeroed which initialises the memory without
Rust seeing it as memory), and that means that the memory is Rust memory and
must adhere to Rust's memory model without data races.

At this point the memory is free to interact with according to normal Rust
rules. To move into the ECMAScript memory model, the `RacyMemory::enter` API
or its siblings must be used.

## Second stage – ECMAScript memory model

Once the `enter` is used, the Rust memory is deallocated (in an abstract sense)
and a new ECMAScript memory slab is allocated in its place, though from the
code point of view it is an entirely new pointer with potentially different
address and provenance. As the Rust memory is now deallocated, using pointers
to read or write into the previous memory area is equivalent to use-after-free
and is thus undefined behaviour.

The newly gained ECMAScript memory in the form of `RacyMemory` can be accessed
using a slice-like data structure called `RacySlice` and its APIs. These APIs
adhere to the ECMAScript memory model, meaning that reads and writes may be
either unordered (or unsynchronising by another name) or sequentially
consistent, and that both atomic and unordered reads and writes may race with
other reads and writes, including ones with mismatching sizes or alignments.

When the ECMAScript memory is no longer needed needed, it can be deallocated
using the `exit` API. This API must not race with any data reads or writes in
the ECMAScript memory (ie. must be synchronised), and produces a new Rust
memory allocation and returns its pointer and length (in case the ECMAScript
memory was slice-like). This memory holds the data that the ECMAScript memory
held at the time of its deallocation and can be used normally.

## Third stage – deallocation

After the `exit` method call, it is up to the caller to deallocate the Rust
memory by using the pointer and length that the `exit` call returns.

# Wait, so you're allocating and deallocating memory at the fences?! That's insanely slow!

Well, no... The fences are inline assembly blocks that take a pointer and
return a pointer, but inside the assembly block is only a comment. The
deallocation and reallocation at the fence is an explanation to the abstract
Rust machine. In truth, no explicit deallocation happens but due to how the
assembly blocks are used (pointer in, pointer out, no
[`pure`](https://doc.rust-lang.org/reference/inline-assembly.html#r-asm.options.supported-options.pure)
option), the compiler has to assume that the allocation may have been
deallocated.

The internals of the library then take care to never perform any reads or
writes through the pointers, nor create any references to non-ZST data from
them. This should keep the compiler from realising that the pointed-to memory
is still actually allocated and useable, as it should since the compiler must
not assume it can eg. inject suprious reads or writes on these pointers.

# Should you use this crate

No, probably not. Unless you're writing a JavaScript, WebAssembly, or Java
virtual machine and you're not willing to emulate the weaker memory model, you
should run away from here and fast. If you are writing a virtual machine for
one of the mentioned languages or some other language with the same memory
model, then first consider the following alternatives:

1. If you're reading this in the future and Rust allows mixed-size atomics,
   then strongly consider using normal Rust atomics with Unordered ordering
   replaced with Relaxed. This gives you the same memory model except for on
   some obscure hardware platforms (if you thought DEC, you're right), and
   perhaps some extra locks on ARM platforms sometimes. That would be my
   choice.
2. Use `AtomicUsize` as your atomic storage and simulate mixed-size atomics by
   performing CAS loops on these usize chunks, and replace Unordered ordering
   with Relaxed.  This may be the smartest option compared, when the other
   option is to put your trust in some random library filled with inline
   assembly copied off the Internet.
3. Use `AtomicU8` as your atomic storage and hope that either the compiler
   takes pity on you and optimises the atomic byte operations into larger
   atomic chunk operations (I am not aware of such optimisations existing), or
   hope your users don't notice that their atomics are not actually atomic and
   can always tear. This is probably a bad idea.

If none of those sounds appetising, then this library is what you want. I'll
even give you a guarantee: ...

*tail lights flicker in the night as the author speeds off*
