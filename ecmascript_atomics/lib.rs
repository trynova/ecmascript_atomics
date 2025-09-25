// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Copyright of the originating code is owned by Firefox authors and Mozilla,
// modifications by Aapo Alasuutari.

//! # Racy atomic operations
//!
//! This library provides atomic operations that match the ECMAScript
//! specification's memory model: this is effectively the same memory model as
//! Java's shared object model. This model allows non-atomic reads and writes
//! to perform data races, and also allows mixed-size atomic reads and writes.
//! Both of these are undefined behaviour in the C++/Rust memory model, which
//! is why these atomic operations are implemented using inline assembly.
//!
//! The contents of the library are copied and adapted from Mozilla Firefox's
//! source code, basing mainly on [GenerateAtomicOperations.py].
//!
//! The fundamental constraints on the functions are:
//!
//! - That any Atomic operations performed on memory used in these
//!   functions must only be accessed using these functions or functions
//!   that have a compatible realisation. Importantly, it is strictly
//!   undefined behaviour to use normal Rust Atomics in conjunction with
//!   the racy atomic operations in this library.
//!
//! - That accesses may race without creating C++/Rust undefined behavior:
//!   atomic accesses (marked "SeqCst") may race with non-atomic
//!   accesses (marked "Unordered"); overlapping but non-matching,
//!   and hence incompatible, atomic accesses may race; and non-atomic
//!   accesses may race. The effects of races need not be predictable,
//!   so garbage can be produced by a read or written by a write, but
//!   the effects must be benign: the program must continue to run, and
//!   only the memory in the union of addresses named in the racing
//!   accesses may be affected.
//!
//! The compatibility constraint means that if the memory is accessed elsewhere
//! then it must make compatible decisions about how to implement atomic
//! operations with the functions below.
//!
//! The safe-for-races constraint means that by and large, it is hard or
//! impossible to implement these primitives in C++/Rust. See "Implementation
//! notes" below.
//!
//! The "SeqCst" suffix on operations means "sequentially consistent"
//! and means such a function's operation must have "sequentially
//! consistent" memory ordering. This corresponds with the Rust Atomics'
//! "SeqCst" ordering.
//!
//! Note that an "Unordered" access does not provide the atomicity of
//! a "relaxed atomic" access: it can read or write garbage if there's
//! a race.
//!
//!
//! ## Implementation notes.
//!
//! It's not a requirement that these functions be inlined.
//!
//! In principle these functions will not be written in Rust, thus
//! making races defined behavior if all racy accesses from Rust go via
//! these functions.
//!
//! The appropriate implementations will be platform-specific and
//! there are some obvious implementation strategies to choose
//! from, sometimes a combination is appropriate:
//!
//!  - generating the code at run-time with a JIT;
//!  - hand-written assembler (maybe inline); or
//!  - using special compiler intrinsics or directives.
//!
//! Trusting the compiler not to generate code that blows up on a
//! race definitely won't work in the presence of TSan, or even of
//! optimizing compilers in seemingly-"innocuous" conditions. (See
//! https://www.usenix.org/legacy/event/hotpar11/tech/final_files/Boehm.pdf
//! for details.)
//!
//! [GenerateAtomicOperations.py]: https://searchfox.org/firefox-main/source/js/src/jit/GenerateAtomicOperations.py

use core::{cell::UnsafeCell, hint::assert_unchecked, marker::PhantomData, ptr::NonNull};
use std::{mem::MaybeUninit, num::NonZeroUsize};

use ecmascript_atomics_gen_copy::gen_copy;

macro_rules! fence {
    (true, x86) => {
        "mfence"
    };
    (true, aarch64) => {
        "dmb ish"
    };
    (true, arm) => {
        "dmb sy"
    };
    (false, $_: tt) => {
        ""
    };
}

macro_rules! gen_load {
    (u8, $ptr: ident, $barrier: tt) => {
        let z: u8;
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "mov {val}, [{ptr}]",
                fence!(false, x86),
                ptr = in(reg) ptr,
                val = lateout(reg_byte) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                "ldrb {val:w}, [{ptr}]",
                fence!($barrier, aarch64),
                ptr = in(reg) ptr,
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            core::arch::asm!(
                "ldrb {val:w}, [{ptr}]",
                fence!($barrier, arm),
                ptr = in(reg) ptr,
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        return z;
    };
    (u16, $ptr: ident, $barrier: tt) => {
        let z: u16;
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "mov {val:x}, [{ptr}]",
                fence!(false, x86),
                ptr = in(reg) ptr,
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                "ldrh {val:w}, [{ptr}]",
                fence!($barrier, aarch64),
                ptr = in(reg) ptr,
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            core::arch::asm!(
                "ldrh {val:w}, [{ptr}]",
                fence!($barrier, arm),
                ptr = in(reg) ptr,
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        return z;
    };
    (u32, $ptr: ident, $barrier: tt) => {
        let z: u32;
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "mov {val:e}, [{ptr}]",
                fence!(false, x86),
                ptr = in(reg) ptr,
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                "ldr {val:w}, [{ptr}]",
                fence!($barrier, aarch64),
                ptr = in(reg) ptr,
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            core::arch::asm!(
                "ldr {val:w}, [{ptr}]",
                fence!($barrier, arm),
                ptr = in(reg) ptr,
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        return z;
    };
    (u64, $ptr: ident, $barrier: tt) => {
        let z: u64;
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::asm!(
                "mov {val:r}, [{ptr}]",
                fence!(false, x86),
                ptr = in(reg) ptr,
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                "ldr {val:x}, [{ptr}]",
                fence!($barrier, aarch64),
                ptr = in(reg) ptr,
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        #[cfg(any(target_arch = "x86", target_arch = "arm"))]
        unsafe {
            const { panic!("Unexpected size") }
        }

        return z;
    };
    ($type: ty, $ptr: ident, $barrier: tt) => {
        panic!("Unsupported type");
    };
}

macro_rules! gen_store {
    (u8, $ptr: ident, $val: ident, $barrier: tt) => {
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "mov [{ptr}], {val}",
                fence!($barrier, x86),
                ptr = in(reg) ptr,
                val = in(reg_byte) $val,
                options(preserves_flags, nostack)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                fence!($barrier, aarch64),
                "strb {val:w}, [{ptr}]",
                fence!($barrier, aarch64),
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(preserves_flags, nostack)
            );
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            core::arch::asm!(
                fence!($barrier, arm),
                "strb {val:w}, [{ptr}]",
                fence!($barrier, arm),
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(preserves_flags, nostack)
            );
        }
    };
    (u16, $ptr: ident, $val: ident, $barrier: tt) => {
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "mov [{ptr}], {val:x}",
                fence!($barrier, x86),
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(preserves_flags, nostack)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                fence!($barrier, aarch64),
                "strh {val:w}, [{ptr}]",
                fence!($barrier, aarch64),
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(preserves_flags, nostack)
            );
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            core::arch::asm!(
                fence!($barrier, arm),
                "strh {val:w}, [{ptr}]",
                fence!($barrier, arm),
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(preserves_flags, nostack)
            );
        }
    };
    (u32, $ptr: ident, $val: ident, $barrier: tt) => {
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "mov [{ptr}], {val:e}",
                fence!($barrier, x86),
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(preserves_flags, nostack)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                fence!($barrier, aarch64),
                "str {val:w}, [{ptr}]",
                fence!($barrier, aarch64),
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(preserves_flags, nostack)
            );
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            core::arch::asm!(
                fence!($barrier, arm),
                "str {val:w}, [{ptr}]",
                fence!($barrier, arm),
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(preserves_flags, nostack)
            );
        }
    };
    (u64, $ptr: ident, $val: ident, $barrier: tt) => {
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::asm!(
                "mov [{ptr}], {val:r}",
                fence!($barrier, x86),
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(preserves_flags, nostack)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                fence!($barrier, aarch64),
                "str {val:x}, [{ptr}]",
                fence!($barrier, aarch64),
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(preserves_flags, nostack)
            );
        }

        #[cfg(any(target_arch = "x86", target_arch = "arm"))]
        unsafe {
            const { panic!("Unexpected size") }
        }
    };
}

macro_rules! gen_exchange {
    (u8, $ptr: ident, $val: ident) => {
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "xchg [{ptr}], {val}",
                ptr = in(reg) ptr,
                val = inout(reg_byte) $val,
                options(preserves_flags, nostack)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let res: u8;
            core::arch::asm!(
                "dmb ish",
                "2:",
                "ldxrb {res:w}, [{ptr}]",
                "stxrb {scratch:w}, {val:w}, [{ptr}]",
                "cbnz {scratch:w}, 2b",
                "dmb ish",
                res = out(reg) res,
                scratch = out(reg) _,
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            let res: u8;
            core::arch::asm!(
                "dmb sy",
                "2:",
                "ldrexb {res:w}, [{ptr}]",
                "strexb {scratch:w}, {val:w}, [{ptr}]",
                "cmp {scratch:w}, #1",
                "beq 2b",
                "dmb sy",
                res = out(reg) res,
                scratch = out(reg) _,
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        return $val;
    };
    (u16, $ptr: ident, $val: ident) => {
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "xchg [{ptr}], {val:x}",
                ptr = in(reg) ptr,
                val = inout(reg) $val,
                options(preserves_flags, nostack)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let res: u16;
            core::arch::asm!(
                "dmb ish",
                "2:",
                "ldxrh {res:w}, [{ptr}]",
                "stxrh {scratch:w}, {val:w}, [{ptr}]",
                "cbnz {scratch:w}, 2b",
                "dmb ish",
                res = out(reg) res,
                scratch = out(reg) _,
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            let res: u16;
            core::arch::asm!(
                "dmb sy",
                "2:",
                "ldrexh {res:w}, [{ptr}]",
                "strexh {scratch:w}, {val:w}, [{ptr}]",
                "cmp {scratch:w}, #1",
                "beq 2b",
                "dmb sy",
                res = out(reg) res,
                scratch = out(reg) _,
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        return $val;
    };
    (u32, $ptr: ident, $val: ident) => {
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "xchg [{ptr}], {val:e}",
                ptr = in(reg) ptr,
                val = inout(reg) $val,
                options(preserves_flags, nostack)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let res: u32;
            core::arch::asm!(
                "dmb ish",
                "2:",
                "ldxr {res:w}, [{ptr}]",
                "stxr {scratch:w}, {val:w}, [{ptr}]",
                "cbnz {scratch:w}, 2b",
                "dmb ish",
                res = out(reg) res,
                scratch = out(reg) _,
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            let res: u32;
            core::arch::asm!(
                "dmb sy",
                "2:",
                "ldrex {res:w}, [{ptr}]",
                "strex {scratch:w}, {val:w}, [{ptr}]",
                "cmp {scratch:w}, #1",
                "beq 2b",
                "dmb sy",
                res = out(reg) res,
                scratch = out(reg) _,
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        return $val;
    };
    (u64, $ptr: ident, $val: ident) => {
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::asm!(
                "xchg [{ptr}], {val:r}",
                ptr = in(reg) ptr,
                val = inout(reg) $val,
                options(preserves_flags, nostack)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let res: u64;
            core::arch::asm!(
                "dmb ish",
                "2:",
                "ldxr {res:x}, [{ptr}]",
                "stxr {scratch:w}, {val:x}, [{ptr}]",
                "cbnz {scratch:w}, 2b",
                "dmb ish",
                res = out(reg) res,
                scratch = out(reg) _,
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        #[cfg(any(target_arch = "x86", target_arch = "arm"))]
        unsafe {
            const { panic!("Unexpected size") }
        }

        return $val;
    };
}

macro_rules! gen_cmpxchg {
    (u8, $ptr: ident, $old_val: ident, $new_val: ident) => {
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "lock; cmpxchg [{ptr}], {new_val}",
                // Load old_val into RAX as input/output register
                inout("al") $old_val,
                ptr = in(reg) ptr,
                new_val = in(reg_byte) $new_val,
                options(nostack)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let res: u8;
            core::arch::asm!(
                "dmb ish",
                "2:",
                "uxtb {scratch:w}, {old_val:w}",
                "ldxrb {res:w}, [{ptr}]",
                "cmp {res:w}, {scratch:w}",
                "b.ne 3f",
                "stxrb {scratch:w}, {new_val:w}, [{ptr}]",
                "cbnz {scratch:w}, 2b",
                "3: dmb ish",
                res = out(reg) res,
                scratch = out(reg) _,
                ptr = in(reg) ptr,
                old_val = in(reg) $old_val,
                new_val = in(reg) $new_val,
                options(nostack)
            );
            $old_val = res;
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            let res: u8;
            core::arch::asm!(
                "dmb sy",
                "2:",
                "uxtb {scratch:w}, {old_val:w}",
                "ldrexb {res:w} [{ptr}]",
                "cmp {res:w}, {scratch:w}",
                "bne 3f",
                "strexb {scratch:w}, {new_val:w}, [{ptr}]",
                "cmp {scratch:w}, #1",
                "beq 2b",
                "3: dmb sy",
                res = out(reg) res,
                scratch = out(reg) _,
                ptr = in(reg) ptr,
                old_val = in(reg) $old_val,
                new_val = in(reg) $new_val,
                options(nostack)
            );
            $old_val = res;
        }

        return $old_val;
    };
    (u16, $ptr: ident, $old_val: ident, $new_val: ident) => {
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "lock; cmpxchg [{ptr}], {new_val:x}",
                // Load old_val into RAX as input/output register
                inout("ax") $old_val,
                ptr = in(reg) ptr,
                new_val = in(reg) $new_val,
                options(nostack)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let res: u16;
            core::arch::asm!(
                "dmb ish",
                "2:",
                "uxth {scratch:w}, {old_val:w}",
                "ldxrh {res:w}, [{ptr}]",
                "cmp {res:w}, {scratch:w}",
                "b.ne 3f",
                "stxrh {scratch:w}, {new_val:w}, [{ptr}]",
                "cbnz {scratch:w}, 2b",
                "3: dmb ish",
                res = out(reg) res,
                scratch = out(reg) _,
                ptr = in(reg) ptr,
                old_val = in(reg) $old_val,
                new_val = in(reg) $new_val,
                options(nostack)
            );
            $old_val = res;
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            let res: u16;
            core::arch::asm!(
                "dmb sy",
                "2:",
                "uxth {scratch}, {old_val}",
                "ldrexh {res} [{ptr}]",
                "cmp {res}, {scratch}",
                "bne 3f",
                "strexh {scratch}, {new_val}, [{ptr}]",
                "cmp {scratch}, #1",
                "beq 2b",
                "3: dmb sy",
                res = out(reg) res,
                scratch = out(reg) _,
                ptr = in(reg) ptr,
                old_val = in(reg) $old_val,
                new_val = in(reg) $new_val,
                options(nostack)
            );
            $old_val = res;
        }

        return $old_val;
    };
    (u32, $ptr: ident, $old_val: ident, $new_val: ident) => {
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "lock; cmpxchg [{ptr}], {new_val:e}",
                // Load old_val into RAX as input/output register
                inout("eax") $old_val,
                ptr = in(reg) ptr,
                new_val = in(reg) $new_val,
                options(nostack)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let res: u32;
            core::arch::asm!(
                "dmb ish",
                "2:",
                "mov {scratch:w}, {old_val:w}",
                "ldxr {res:w}, [{ptr}]",
                "cmp {res:w}, {scratch:w}",
                "b.ne 3f",
                "stxr {scratch:w}, {new_val:w}, [{ptr}]",
                "cbnz {scratch:w}, 2b",
                "3: dmb ish",
                res = out(reg) res,
                scratch = out(reg) _,
                ptr = in(reg) ptr,
                old_val = in(reg) $old_val,
                new_val = in(reg) $new_val,
                options(nostack)
            );
            $old_val = res;
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            let res: u32;
            core::arch::asm!(
                "dmb sy",
                "2:",
                "mov {scratch}, {old_val}",
                "ldrex {res} [{ptr}]",
                "cmp {res}, {scratch}",
                "bne 3f",
                "strex {scratch}, {new_val}, [{ptr}]",
                "cmp {scratch}, #1",
                "beq 2b",
                "3: dmb sy",
                res = out(reg) res,
                scratch = out(reg) _,
                ptr = in(reg) ptr,
                old_val = in(reg) $old_val,
                new_val = in(reg) $new_val,
                options(nostack)
            );
            $old_val = res;
        }

        return $old_val;
    };
    (u64, $ptr: ident, $old_val: ident, $new_val: ident) => {
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(target_arch = "x86")]
        unsafe {
            let [b0, b1, b2, b3, b4, b5, b6, b7] = $old_val.to_le_bytes();
            let old_bot = u32::from_le_bytes([b0, b1, b2, b3]);
            let old_top = u32::from_le_bytes([b4, b5, b6, b7]);
            let [b0, b1, b2, b3, b4, b5, b6, b7] = $new_val.to_le_bytes();
            let new_bot = u32::from_le_bytes([b0, b1, b2, b3]);
            let new_top = u32::from_le_bytes([b4, b5, b6, b7]);
            core::arch::asm!(
                "lock; cmpxchg8b [{ptr}]",
                // Load old_val into EDX:EAX (high:low).
                inout("edx") old_top,
                inout("eax") old_bot,
                ptr = in(reg) ptr,
                // Load old_val into ECX:EBX (high:low).
                in("ecx") new_top,
                in("ebx") new_bot,
                options(nostack)
            );
            let [b0, b1, b2, b3] = old_bot.to_le_bytes();
            let [b4, b5, b6, b7] = old_top.to_le_bytes();
            $old_val = u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]);
        }

        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::asm!(
                "lock; cmpxchg [{ptr}], {new_val:r}",
                // Load old_val into RAX as input/output register
                inout("rax") $old_val,
                ptr = in(reg) ptr,
                new_val = in(reg) $new_val,
                options(nostack)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let res: u64;
            core::arch::asm!(
                "dmb ish",
                "2:",
                "mov {scratch:x}, {old_val:x}",
                "ldxr {res:x}, [{ptr}]",
                "cmp {res:x}, {scratch:x}",
                "b.ne 3f",
                "stxr {scratch:w}, {new_val:x}, [{ptr}]",
                "cbnz {scratch:w}, 2b",
                "3: dmb ish",
                res = out(reg) res,
                scratch = out(reg) _,
                ptr = in(reg) ptr,
                old_val = in(reg) $old_val,
                new_val = in(reg) $new_val,
                options(nostack)
            );
            $old_val = res;
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            let [b0, b1, b2, b3, b4, b5, b6, b7] = $old_val.to_le_bytes();
            let old_bot = u32::from_le_bytes([b0, b1, b2, b3]);
            let old_top = u32::from_le_bytes([b4, b5, b6, b7]);
            let [b0, b1, b2, b3, b4, b5, b6, b7] = $new_val.to_le_bytes();
            let new_bot = u32::from_le_bytes([b0, b1, b2, b3]);
            let new_top = u32::from_le_bytes([b4, b5, b6, b7]);
            core::arch::asm!(
                "dmb sy",
                "2: ldrexd r0 r1 [{ptr}]",
                "cmp r0 {old_bot}",
                "b.ne 3f",
                "cmp r1 {old_top}",
                "b.ne 3f",
                "mov r2, {new_bot}"
                "mov r3, {new_top}"
                "strexd r4, r2, r3, [{ptr}]"
                "cmp r4, #1",
                "beq 2b",
                "3: dmb sy",
                "mov {old_bot} r0",
                "mov {old_top} r1",
                inout(reg) old_bot,
                inout(reg) old_top,
                ptr = in(reg) ptr,
                new_bot = in(reg) new_bot,
                new_top = in(reg) new_top,
                out("r0") _,
                out("r1") _,
                out("r2") _,
                out("r3") _,
                out("r4") _,
                options(nostack)
            );
            let [b0, b1, b2, b3] = old_bot.to_le_bytes();
            let [b4, b5, b6, b7] = old_top.to_le_bytes();
            $old_val = u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]);
        }

        return $old_val;
    };
}

macro_rules! fetchop {
    // The `add` operation can be optimized with XADD.
    ("add", x86, u8) => {
        "lock; xadd [{ptr}], {val}"
    };
    ("add", x86, u16) => {
        "lock; xadd [{ptr}], {val:x}"
    };
    ("add", x86, u32) => {
        "lock; xadd [{ptr}], {val:e}"
    };
    ("add", x86, u64) => {
        "lock; xadd [{ptr}], {val:r}"
    };
    ("and", x86, u8) => {
        "and {scratch}, {val}"
    };
    ("and", x86, u16) => {
        "and {scratch:x}, {val:x}"
    };
    ("and", x86, u32) => {
        "and {scratch:e}, {val:e}"
    };
    ("and", x86, u64) => {
        "and {scratch:r}, {val:r}"
    };
    ("or", x86, u8) => {
        "or {scratch}, {val}"
    };
    ("or", x86, u16) => {
        "or {scratch:x}, {val:x}"
    };
    ("or", x86, u32) => {
        "or {scratch:e}, {val:e}"
    };
    ("or", x86, u64) => {
        "or {scratch:r}, {val:r}"
    };
    ("xor", x86, u8) => {
        "xor {scratch}, {val}"
    };
    ("xor", x86, u16) => {
        "xor {scratch:x}, {val:x}"
    };
    ("xor", x86, u32) => {
        "xor {scratch:e}, {val:e}"
    };
    ("xor", x86, u64) => {
        "xor {scratch:r}, {val:r}"
    };
    // Note: we differ here from source material. In Firefox the operation
    // always operates on :x registers; there doesn't seem to be a reason for
    // this so we try to avoid that.
    ("add", aarch64, u32) => {
        "add {scratch1:w}, {res:w}, {val:w}"
    };
    ("add", aarch64, u64) => {
        "add {scratch1:x}, {res:x}, {val:x}"
    };
    ("and", aarch64, u32) => {
        "and {scratch1:w}, {res:w}, {val:w}"
    };
    ("and", aarch64, u64) => {
        "and {scratch1:x}, {res:x}, {val:x}"
    };
    ("or", aarch64, u32) => {
        "orr {scratch1:w}, {res:w}, {val:w}"
    };
    ("or", aarch64, u64) => {
        "orr {scratch1:x}, {res:x}, {val:x}"
    };
    ("xor", aarch64, u32) => {
        "eor {scratch1:w}, {res:w}, {val:w}"
    };
    ("xor", aarch64, u64) => {
        "eor {scratch1}, {res}, {val}"
    };
    ("add", arm) => {
        "add {scratch1}, {res}, {val}"
    };
    ("and", arm) => {
        "and {scratch1}, {res}, {val}"
    };
    ("or", arm) => {
        "orr {scratch1}, {res}, {val}"
    };
    ("xor", arm) => {
        "eor {scratch1}, {res}, {val}"
    };
}

macro_rules! gen_fetchop {
    (u8, $op: tt, $ptr: ident, $val: ident) => {
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if $op == "add" {
                // The `add` operation can be optimized with XADD.
                core::arch::asm!(
                    "lock; xadd [{ptr}], {val}",
                    val = inout(reg_byte) $val,
                    ptr = in(reg) ptr,
                    options(nostack)
                );
            } else {
                let res: u8;
                core::arch::asm!(
                    "mov al, [{ptr}]",
                    "2: mov {scratch}, al",
                    fetchop!($op, x86, u8),
                    "lock; cmpxchg [{ptr}], {scratch}",
                    "jnz 2b",
                    // Use of RAX is required for the CMPXCHG instruction.
                    out("al") res,
                    scratch = out(reg_byte) _,
                    ptr = in(reg) ptr,
                    val = in(reg_byte) $val,
                    options(nostack)
                );
                $val = res;
            }
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let res: u8;
            core::arch::asm!(
                "dmb ish",
                "2:",
                "ldxrb {res:w}, [{ptr}]",
                fetchop!($op, aarch64, u32),
                "stxrb {scratch2:w}, {scratch1:w}, [{ptr}]",
                "cbnz {scratch2:w}, 2b",
                "3: dmb ish",
                res = out(reg) res,
                scratch1 = out(reg) _,
                scratch2 = out(reg) _,
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            let res: u8;
            core::arch::asm!(
                "dmb sy",
                "2:",
                "ldrexb {res}, [{ptr}]",
                fetchop!($op, arm),
                "strexb {scratch2}, {scratch1}, [{ptr}]",
                "cmp {scratch2}, #1",
                "beq 2b",
                "dmb sy",
                res = out(reg) res,
                scratch1 = out(reg) _,
                scratch2 = out(reg) _,
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        return $val;
    };
    (u16, $op: tt, $ptr: ident, $val: ident) => {
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if $op == "add" {
                // The `add` operation can be optimized with XADD.
                core::arch::asm!(
                    "lock; xadd [{ptr}], {val:x}",
                    val = inout(reg) $val,
                    ptr = in(reg) ptr,
                    options(nostack)
                );
            } else {
                let res: u16;
                core::arch::asm!(
                    "mov ax, [{ptr}]",
                    "2: mov {scratch:x}, ax",
                    fetchop!($op, x86, u16),
                    "lock; cmpxchg [{ptr}], {scratch:x}",
                    "jnz 2b",
                    // Use of RAX is required for the CMPXCHG instruction.
                    out("ax") res,
                    scratch = out(reg) _,
                    ptr = in(reg) ptr,
                    val = in(reg) $val,
                    options(nostack)
                );
                $val = res;
            }
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let res: u16;
            core::arch::asm!(
                "dmb ish",
                "2:",
                "ldxrh {res:w}, [{ptr}]",
                fetchop!($op, aarch64, u32),
                "stxrh {scratch2:w}, {scratch1:w}, [{ptr}]",
                "cbnz {scratch2:w}, 2b",
                "3: dmb ish",
                res = out(reg) res,
                scratch1 = out(reg) _,
                scratch2 = out(reg) _,
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            let res: u16;
            core::arch::asm!(
                "dmb sy",
                "2:",
                "ldrexh {res}, [{ptr}]",
                fetchop!($op, arm),
                "strexh {scratch2}, {scratch1}, [{ptr}]",
                "cmp {scratch2}, #1",
                "beq 2b",
                "dmb sy",
                res = out(reg) res,
                scratch1 = out(reg) _,
                scratch2 = out(reg) _,
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        return $val;
    };
    (u32, $op: tt, $ptr: ident, $val: ident) => {
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if $op == "add" {
                // The `add` operation can be optimized with XADD.
                core::arch::asm!(
                    "lock; xadd [{ptr}], {val:e}",
                    val = inout(reg) $val,
                    ptr = in(reg) ptr,
                    options(nostack)
                );
            } else {
                let res: u32;
                core::arch::asm!(
                    "mov eax, [{ptr}]",
                    "2: mov {scratch:e}, eax",
                    fetchop!($op, x86, u32),
                    "lock; cmpxchg [{ptr}], {scratch:e}",
                    "jnz 2b",
                    // Use of RAX is required for the CMPXCHG instruction.
                    out("eax") res,
                    scratch = out(reg) _,
                    ptr = in(reg) ptr,
                    val = in(reg) $val,
                    options(nostack)
                );
                $val = res;
            }
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let res: u32;
            core::arch::asm!(
                "dmb ish",
                "2:",
                "ldxr {res:w}, [{ptr}]",
                fetchop!($op, aarch64, u32),
                "stxr {scratch2:w}, {scratch1:w}, [{ptr}]",
                "cbnz {scratch2:w}, 2b",
                "3: dmb ish",
                res = out(reg) res,
                scratch1 = out(reg) _,
                scratch2 = out(reg) _,
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            let res: u32;
            core::arch::asm!(
                "dmb sy",
                "2:",
                "ldrex {res}, [{ptr}]",
                fetchop!($op, arm),
                "strex {scratch2}, {scratch1}, [{ptr}]",
                "cmp {scratch2}, #1",
                "beq 2b",
                "dmb sy",
                res = out(reg) res,
                scratch1 = out(reg) _,
                scratch2 = out(reg) _,
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        return $val;
    };
    (u64, $op: tt, $ptr: ident, $val: ident) => {
        // SAFETY: ptr is NonNull<()>; it is never null, dangling, or unaligned.
        let ptr = unsafe { &mut *$ptr.as_ptr() };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if $op == "add" {
                // The `add` operation can be optimized with XADD.
                core::arch::asm!(
                    "lock; xadd [{ptr}], {val:r}",
                    val = inout(reg) $val,
                    ptr = in(reg) ptr,
                    options(nostack)
                );
            } else {
                let res: u64;
                core::arch::asm!(
                    "mov rax, [{ptr}]",
                    "2: mov {scratch:r}, rax",
                    fetchop!($op, x86, u64),
                    "lock; cmpxchg [{ptr}], {scratch:r}",
                    "jnz 2b",
                    // Use of RAX is required for the CMPXCHG instruction.
                    out("rax") res,
                    scratch = out(reg) _,
                    ptr = in(reg) ptr,
                    val = in(reg) $val,
                    options(nostack)
                );
                $val = res;
            }
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let res: u64;
            core::arch::asm!(
                "dmb ish",
                "2:",
                "ldxr {res:x}, [{ptr}]",
                fetchop!($op, aarch64, u64),
                "stxr {scratch2:w}, {scratch1:x}, [{ptr}]",
                "cbnz {scratch2:w}, 2b",
                "3: dmb ish",
                res = out(reg) res,
                scratch1 = out(reg) _,
                scratch2 = out(reg) _,
                ptr = in(reg) ptr,
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            const { panic!("Unexpected size") }
        }

        return $val;
    };
}

/// ECMAScript atomic memory orderings
///
/// Memory orderings specify the way atomic operations synchronise memory.
/// With [`Ordering::Unordered`], no synchronisation is performed. With
/// [`Ordering::SeqCst`], a store-load pair of operations synchronize other
/// memory while additionally preserving a total order of such operations
/// across all threads.
///
/// The ECMAScript memory model is explained in the [ECMAScript Language
/// specification](https://tc39.es/ecma262/#sec-memory-model). Note that the
/// "INIT" ordering is not offered here as it is the purview of the memory
/// allocator.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Ordering {
    Unordered,
    SeqCst,
}

/// A sequentially consistent atomic fence.
///
/// See [std::sync::atomic::fence] for details.
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub fn fence() {
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_load_8_seq_cst(ptr: NonNull<()>) -> u8 {
    gen_load!(u8, ptr, true);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_load_16_seq_cst(ptr: NonNull<()>) -> u16 {
    gen_load!(u16, ptr, true);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_load_32_seq_cst(ptr: NonNull<()>) -> u32 {
    gen_load!(u32, ptr, true);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
fn atomic_load_64_seq_cst(ptr: NonNull<()>) -> u64 {
    gen_load!(u64, ptr, true);
}

// These are access-atomic up to sizeof(uintptr_t).
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_load_8_unsynchronized(ptr: NonNull<()>) -> u8 {
    gen_load!(u8, ptr, false);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_load_16_unsynchronized(ptr: NonNull<()>) -> u16 {
    gen_load!(u16, ptr, false);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_load_32_unsynchronized(ptr: NonNull<()>) -> u32 {
    gen_load!(u32, ptr, false);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
fn atomic_load_64_unsynchronized(ptr: NonNull<()>) -> u64 {
    gen_load!(u64, ptr, false);
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_store_8_seq_cst(ptr: NonNull<()>, val: u8) {
    gen_store!(u8, ptr, val, true);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_store_16_seq_cst(ptr: NonNull<()>, val: u16) {
    gen_store!(u16, ptr, val, true);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_store_32_seq_cst(ptr: NonNull<()>, val: u32) {
    gen_store!(u32, ptr, val, true);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
fn atomic_store_64_seq_cst(ptr: NonNull<()>, val: u64) {
    gen_store!(u64, ptr, val, true);
}

// These are access-atomic up to sizeof(uintptr_t).
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_store_8_unsynchronized(ptr: NonNull<()>, val: u8) {
    gen_store!(u8, ptr, val, false);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_store_16_unsynchronized(ptr: NonNull<()>, val: u16) {
    gen_store!(u16, ptr, val, false);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_store_32_unsynchronized(ptr: NonNull<()>, val: u32) {
    gen_store!(u32, ptr, val, false);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
fn atomic_store_64_unsynchronized(ptr: NonNull<()>, val: u64) {
    gen_store!(u64, ptr, val, false);
}

// `exchange` takes a cell address and a value. It stores it in the cell and
// returns the value previously in the cell.
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_exchange_8_seq_cst(ptr: NonNull<()>, mut val: u8) -> u8 {
    gen_exchange!(u8, ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_exchange_16_seq_cst(ptr: NonNull<()>, mut val: u16) -> u16 {
    gen_exchange!(u16, ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_exchange_32_seq_cst(ptr: NonNull<()>, mut val: u32) -> u32 {
    gen_exchange!(u32, ptr, val);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
fn atomic_exchange_64_seq_cst(ptr: NonNull<()>, mut val: u64) -> u64 {
    gen_exchange!(u64, ptr, val);
}

// `cmpxchg` takes a cell address, an expected value and a replacement value.
// If the value in the cell equals the expected value then the replacement value
// is stored in the cell. It always returns the value previously in the cell.
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_cmp_xchg_8_seq_cst(ptr: NonNull<()>, mut old_val: u8, new_val: u8) -> u8 {
    gen_cmpxchg!(u8, ptr, old_val, new_val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_cmp_xchg_16_seq_cst(ptr: NonNull<()>, mut old_val: u16, new_val: u16) -> u16 {
    gen_cmpxchg!(u16, ptr, old_val, new_val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_cmp_xchg_32_seq_cst(ptr: NonNull<()>, mut old_val: u32, new_val: u32) -> u32 {
    gen_cmpxchg!(u32, ptr, old_val, new_val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_cmp_xchg_64_seq_cst(ptr: NonNull<()>, mut old_val: u64, new_val: u64) -> u64 {
    gen_cmpxchg!(u64, ptr, old_val, new_val);
}

// `add` adds a value atomically to the cell and returns the old value in the
// cell. (There is no `sub`; just add the negated value.)
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_add_8_seq_cst(ptr: NonNull<()>, mut val: u8) -> u8 {
    gen_fetchop!(u8, "add", ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_add_16_seq_cst(ptr: NonNull<()>, mut val: u16) -> u16 {
    gen_fetchop!(u16, "add", ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_add_32_seq_cst(ptr: NonNull<()>, mut val: u32) -> u32 {
    gen_fetchop!(u32, "add", ptr, val);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
fn atomic_add_64_seq_cst(ptr: NonNull<()>, mut val: u64) -> u64 {
    gen_fetchop!(u64, "add", ptr, val);
}

// `and` bitwise-and a value atomically into the cell and returns the old value
// in the cell.
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_and_8_seq_cst(ptr: NonNull<()>, mut val: u8) -> u8 {
    gen_fetchop!(u8, "and", ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_and_16_seq_cst(ptr: NonNull<()>, mut val: u16) -> u16 {
    gen_fetchop!(u16, "and", ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_and_32_seq_cst(ptr: NonNull<()>, mut val: u32) -> u32 {
    gen_fetchop!(u32, "and", ptr, val);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
fn atomic_and_64_seq_cst(ptr: NonNull<()>, mut val: u64) -> u64 {
    gen_fetchop!(u64, "and", ptr, val);
}

// `or` bitwise-ors a value atomically into the cell and returns the old value
// in the cell.
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_or_8_seq_cst(ptr: NonNull<()>, mut val: u8) -> u8 {
    gen_fetchop!(u8, "or", ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_or_16_seq_cst(ptr: NonNull<()>, mut val: u16) -> u16 {
    gen_fetchop!(u16, "or", ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_or_32_seq_cst(ptr: NonNull<()>, mut val: u32) -> u32 {
    gen_fetchop!(u32, "or", ptr, val);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
fn atomic_or_64_seq_cst(ptr: NonNull<()>, mut val: u64) -> u64 {
    gen_fetchop!(u64, "or", ptr, val);
}

// `xor` bitwise-xors a value atomically into the cell and returns the old value
// in the cell.
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_xor_8_seq_cst(ptr: NonNull<()>, mut val: u8) -> u8 {
    gen_fetchop!(u8, "xor", ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_xor_16_seq_cst(ptr: NonNull<()>, mut val: u16) -> u16 {
    gen_fetchop!(u16, "xor", ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_xor_32_seq_cst(ptr: NonNull<()>, mut val: u32) -> u32 {
    gen_fetchop!(u32, "xor", ptr, val);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
fn atomic_xor_64_seq_cst(ptr: NonNull<()>, mut val: u64) -> u64 {
    gen_fetchop!(u64, "xor", ptr, val);
}

/// Emits a machine instruction to signal the processor that it is running in a
/// busy-wait spin-loop (“spin lock”).
///
/// See [std::hint::spin_loop] for details.
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub fn atomic_pause() {
    core::hint::spin_loop();
}

/// Size of a word (pointer) on this architecture.
const WORD_SIZE: usize = size_of::<usize>();
/// Number of words in a block (~cache line) on this architecture. Known to be
/// 8 on all supported architectures.
const WORDS_IN_BLOCK: usize = 8;
/// Size of a block (~cache line) on this architecture in bytes.
const BLOCK_SIZE: usize = WORD_SIZE * WORDS_IN_BLOCK;

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_copy_unaligned_block_down_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(u8, BLOCK_SIZE, "down");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_copy_unaligned_block_up_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(u8, BLOCK_SIZE, "up");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_copy_unaligned_word_down_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(u8, WORD_SIZE, "down");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_copy_unaligned_word_up_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(u8, WORD_SIZE, "up");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_copy_block_down_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(usize, WORDS_IN_BLOCK, "down");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_copy_block_up_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(usize, WORDS_IN_BLOCK, "up");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_copy_word_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(usize, 1, "down");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_copy32_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(u32, 1, "down");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_copy16_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(u16, 1, "down");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
fn atomic_copy8_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(u8, 1, "down");
}

/// Opaque handle to a slice of memory with the ECMAScript Atomics memory
/// model. A slice must be created using the [`enter`] method and must be
/// turned back into Rust memory using the [`exit`] method (note; this must be
/// strictly synchronised with all possible users of the racy atomic memory).
///
/// [`enter`]: crate::RacyAtomicSlice::enter
/// [`exit`]: crate::RacyAtomicSlice::exit
///
/// # Soundness
///
/// The memory behind this handle is not and must not be read as Rust memory.
/// Any Rust reads or writes into the memory, even ones in unreachable blocks,
/// are undefined behaviour.
///
/// # Allocations
///
/// The [`enter`] method takes ownership of Rust memory and (conceptually)
/// deallocates it. Therefore, the pointer passed into the method must not be
/// deallocated by the caller. Conversely, the [`exit`] method (conceptually)
/// allocates new Rust memory and returns a pointer to it to the caller. The
/// caller is then responsible for deallocating the Rust memory.
///
/// Each `enter` call must be matched by an equal `exit` call on the same
/// slice, lest the ECMAScript memory be leaked.
///
/// [`enter`]: crate::RacyAtomicSlice::enter
/// [`exit`]: crate::RacyAtomicSlice::exit
pub struct RacyAtomicSlice<'a> {
    ptr: NonNull<()>,
    len: usize,
    __marker: PhantomData<&'a UnsafeCell<u8>>,
}

/// On x86, unaligned accesses work fine.
const UNALIGNED_ACCESS_IS_OK: bool = cfg!(any(target_arch = "x86", target_arch = "x86_64"));

impl<'a> RacyAtomicSlice<'a> {
    /// Move a memory allocation into the ECMAScript memory model.
    ///
    /// # Safety
    ///
    /// `ptr` must be a pointer to `len` bytes of readable and writable memory.
    ///
    /// # Soundness
    ///
    /// This deallocates the existing memory behind `ptr` (in an abstract
    /// machine sense), such that after this call the `ptr` must be viewed as
    /// an invalid pointer. It is thus strictly forbidden for any Rust code to
    /// read from or write into the memory, including with atomic operations.
    #[inline]
    #[must_use]
    pub unsafe fn enter(ptr: NonNull<u8>, len: usize) -> Self {
        let mut ptr = ptr.as_ptr();
        // SAFETY: noop.
        unsafe {
            core::arch::asm!(
                "/* Magic spell: let {} no longer be memory in Rust's eyes! */",
                // Note: ptr is and out parameter so that the assembly block
                // can conceptually deallocate the original ptr, removing its
                // provenance, and return a new ptr with difference provenance.
                inlateout(reg) ptr,
                options(nostack, preserves_flags)
            )
        }
        Self {
            // SAFETY: Magic spell always returns non-null pointers.
            ptr: unsafe { NonNull::new_unchecked(ptr.cast()) },
            len,
            __marker: PhantomData,
        }
    }

    /// Move an ECMAScript memory model into the void and return a new Rust
    /// memory pointer to `len` bytes.
    ///
    /// # Safety
    ///
    /// This must be the only referrer to the racy atomic memory.
    ///
    /// # Soundness
    ///
    /// This deallocates the ECMAScript memory behind `ptr` (in an abstract
    /// sense). It is thus strictly forbidden to use any RacyAtomics derived
    /// from this memory after this call happens, on any thread. This call must
    /// therefore be strictly synchronised between threads, and only one thread
    /// is allowed to perform this call.
    #[inline]
    #[must_use]
    pub unsafe fn exit(self) -> (NonNull<u8>, usize) {
        let mut ptr = self.ptr.as_ptr();
        // SAFETY: noop.
        unsafe {
            core::arch::asm!(
                "/* Magic spell: let {} be memory in Rust's eyes! */",
                // Note: ptr is and out parameter so that the assembly block
                // can conceptually allocate new memory and return a new ptr
                // into it.
                inlateout(reg) ptr,
                options(nostack, preserves_flags)
            )
        }
        // SAFETY: Magic spell always returns non-null pointers.
        (unsafe { NonNull::new_unchecked(ptr.cast()) }, self.len)
    }

    /// Destructure a racy atomic memory slice into raw parts.
    #[inline(always)]
    pub fn into_raw_parts(self) -> (RacyAtomicPtr, usize) {
        (RacyAtomicPtr::from_ptr(self.ptr), self.len)
    }

    /// Create a new racy atomic memory slice from a racy atomic byte pointer
    /// and length.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a racy atomic memory with a length of at least
    /// `len` bytes. See [new] for the soundness requirements of racy atomic
    /// memory. Note that this function does not "reallocate" the pointed-to memory.
    #[inline(always)]
    pub unsafe fn from_raw_parts(ptr: RacyAtomicPtr, len: usize) -> Self {
        Self {
            ptr: ptr.0,
            len,
            __marker: PhantomData,
        }
    }

    /// Returns the number of bytes in the slice.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the slice has a length of 0.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if the slice is correctly aligned for type `T`.
    ///
    /// Generally you'll only want to call this with the supported API types
    /// `u8`, `u16`, `u32`, and `u64`.
    #[inline(always)]
    pub fn is_aligned<T>(&self) -> bool {
        self.ptr.cast::<T>().is_aligned()
    }

    /// Create a slice of racy atomic memory starting at the given offset.
    /// Returns an empty slice if the offset is beyond the end of this slice.
    #[inline]
    pub fn slice_from(&self, offset: usize) -> Self {
        Self {
            // SAFETY: cannot overflow len.
            ptr: unsafe { self.ptr.byte_add(offset.min(self.len)) },
            len: self.len.saturating_sub(offset),
            __marker: PhantomData,
        }
    }

    /// Create a slice of racy atomic memory ending at the given offset.
    /// Returns self if the offset is beyond the end of this slice.
    #[inline]
    pub fn slice_to(&self, offset: usize) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len.min(offset),
            __marker: PhantomData,
        }
    }

    /// Create a slice of racy atomic memory with the given offsets, bounded to
    /// the end of this slice.
    #[inline]
    pub fn slice(&self, start: usize, end: usize) -> Self {
        // First bound the offsets within the slice.
        let start = start.min(self.len);
        let end = end.min(self.len);
        // Then calculate the slice length, saturating at 0 if order is reversed.
        let len = end.saturating_sub(start);
        Self {
            // SAFETY: cannot overflow len.
            ptr: unsafe { self.ptr.byte_add(start) },
            len,
            __marker: PhantomData,
        }
    }

    /// Check if this slice of racy atomic memory overlaps with another slice.
    #[inline]
    pub fn overlaps_with(&self, other: &Self) -> bool {
        // SAFETY: ptr is guaranteed to point to at least len bytes of valid memory.
        unsafe {
            self.ptr <= other.ptr && other.ptr < self.ptr.byte_add(self.len)
                || other.ptr <= self.ptr && self.ptr < other.ptr.byte_add(other.len)
        }
    }

    /// Load a u8 from the start of this slice using Unordered atomic ordering.
    ///
    /// Returns None if the slice is empty.
    #[inline]
    pub fn load_u8(&self) -> Option<u8> {
        if self.is_empty() {
            return None;
        }
        let mut scratch = MaybeUninit::<u8>::uninit();
        let dst = unsafe { NonNull::new_unchecked(scratch.as_mut_ptr().cast::<()>()) };
        // SAFETY: stack pointer cannot be zero.
        atomic_copy8_unsynchronized(self.ptr, dst);
        // SAFETY: copy has initialised the scratch data.
        Some(unsafe { scratch.assume_init() })
    }

    /// Load a u16 from the start of this slice using Unordered atomic
    /// ordering. The slice need not be aligned to the type.
    ///
    /// Returns None if the slice is smaller than 2 bytes.
    ///
    /// # Tearing
    ///
    /// This load may tear
    #[inline]
    pub fn load_u16(&self) -> Option<u16> {
        if self.len() < 2 {
            return None;
        }
        let mut scratch = MaybeUninit::<u16>::uninit();
        let dst = unsafe { NonNull::new_unchecked(scratch.as_mut_ptr().cast::<()>()) };
        // SAFETY: checked self length, dst is proper length.
        unsafe {
            atomic_copy_down_no_tear_if_aligned_unsynchronized(self.ptr, dst, self.ptr.byte_add(2))
        };
        // SAFETY: copy has initialised the scratch data.
        Some(unsafe { scratch.assume_init() })
    }

    /// Load a u32 from the start of this slice using Unordered atomic
    /// ordering. The slice need not be aligned to the type.
    ///
    /// Returns None if the slice is smaller than 4 bytes.
    ///
    /// # Tearing
    ///
    /// This load may tear
    #[inline]
    pub fn load_u32(&self) -> Option<u32> {
        if self.len() < 4 {
            return None;
        }
        let mut scratch = MaybeUninit::<u32>::uninit();
        let dst = unsafe { NonNull::new_unchecked(scratch.as_mut_ptr().cast::<()>()) };
        // SAFETY: checked self length, dst is proper length.
        unsafe {
            atomic_copy_down_no_tear_if_aligned_unsynchronized(self.ptr, dst, self.ptr.byte_add(4))
        };
        // SAFETY: copy has initialised the scratch data.
        Some(unsafe { scratch.assume_init() })
    }

    /// Load a u64 from the start of this slice using Unordered atomic
    /// ordering. The slice need not be aligned to the type.
    ///
    /// Returns None if the slice is smaller than 8 bytes.
    ///
    /// # Tearing
    ///
    /// This load may tear
    #[inline]
    pub fn load_u64(&self) -> Option<u64> {
        if self.len() < 8 {
            return None;
        }
        let mut scratch = MaybeUninit::<u64>::uninit();
        let dst = unsafe { NonNull::new_unchecked(scratch.as_mut_ptr().cast::<()>()) };
        // SAFETY: checked self length, dst is proper length.
        unsafe { unordered_memcpy_down_unsynchronized(self.ptr, dst, 8) };
        // SAFETY: copy has initialised the scratch data.
        Some(unsafe { scratch.assume_init() })
    }

    /// Store a u8 into the start of this slice using Unordered atomic ordering
    /// mode.
    ///
    /// Returns None if the slice is empty.
    #[inline]
    pub fn store_u8(&self, val: u8) -> Option<()> {
        if self.is_empty() {
            return None;
        }
        // SAFETY: stack pointer cannot be zero.
        let src = unsafe { NonNull::new_unchecked(&val as *const _ as *mut ()) };
        atomic_copy8_unsynchronized(src, self.ptr);
        Some(())
    }

    /// Store a u16 into the start of this slice using Unordered atomic
    /// ordering mode.
    ///
    /// Returns None if the slice is smaller than 2 bytes.
    ///
    /// # Tearing
    ///
    /// This load may tear
    #[inline]
    pub fn store_u16(&self, val: u16) -> Option<()> {
        if self.len() < 2 {
            return None;
        }
        // SAFETY: checked self length, dst is proper length.
        let src = unsafe { NonNull::new_unchecked(&val as *const _ as *mut ()) };
        unsafe { unordered_memcpy_down_unsynchronized(src, self.ptr, 4) };
        Some(())
    }

    /// Store a u32 into the start of this slice using Unordered atomic
    /// ordering mode.
    ///
    /// Returns None if the slice is smaller than 4 bytes.
    ///
    /// # Tearing
    ///
    /// This load may tear
    #[inline]
    pub fn store_u32(&self, val: u32) -> Option<()> {
        if self.len() < 4 {
            return None;
        }
        // SAFETY: checked self length, dst is proper length.
        let src = unsafe { NonNull::new_unchecked(&val as *const _ as *mut ()) };
        unsafe { unordered_memcpy_down_unsynchronized(src, self.ptr, 4) };
        Some(())
    }

    /// Store a u64 into the start of this slice using Unordered atomic
    /// ordering mode.
    ///
    /// Returns None if the slice is smaller than 8 bytes.
    ///
    /// # Tearing
    ///
    /// This load may tear
    #[inline]
    pub fn store_u64(&self, val: u64) -> Option<()> {
        if self.len() < 8 {
            return None;
        }
        // SAFETY: checked self length, dst is proper length.
        let src = unsafe { NonNull::new_unchecked(&val as *const _ as *mut ()) };
        unsafe { unordered_memcpy_down_unsynchronized(src, self.ptr, 8) };
        Some(())
    }

    /// Convert the memory into a racy atomic byte at the start of the slice.
    /// Returns None if the slice is smaller than 1 byte.
    pub fn as_u8(&self) -> Option<RacyAtomicU8> {
        if self.is_empty() {
            None
        } else {
            Some(RacyAtomicU8::from_ptr(self.ptr))
        }
    }

    /// Convert the memory into a racy atomic u16 at the start of the slice.
    /// Returns None if the slice is smaller than 2 bytes in size or is not
    /// correctly aligned.
    pub fn as_u16(&self) -> Option<RacyAtomicU16> {
        if self.len() < 2 || !self.ptr.cast::<u16>().is_aligned() {
            None
        } else {
            Some(RacyAtomicU16::from_ptr(self.ptr))
        }
    }

    /// Convert the memory into a racy atomic u32 at the start of the slice.
    /// Returns None if the slice is smaller than 4 bytes in size or is not
    /// correctly aligned.
    pub fn as_u32(&self) -> Option<RacyAtomicU32> {
        if self.len() < 4 || !self.ptr.cast::<u32>().is_aligned() {
            None
        } else {
            Some(RacyAtomicU32::from_ptr(self.ptr))
        }
    }

    /// Convert the memory into a racy atomic u64 at the start of the slice.
    /// Returns None if the slice is smaller than 8 bytes in size or is not
    /// correctly aligned.
    pub fn as_u64(&self) -> Option<RacyAtomicU64> {
        if self.len() < 8 || !self.ptr.cast::<u64>().is_aligned() {
            None
        } else {
            Some(RacyAtomicU64::from_ptr(self.ptr))
        }
    }
}

/// Opaque pointer to memory in the ECMAScript racy atomics memory model.
///
/// This is intended for unsafe usage only, where eg. the size of an EMCAScript
/// memory is stored separately from the pointer.
pub struct RacyAtomicPtr(NonNull<()>);

impl RacyAtomicPtr {
    fn from_ptr(ptr: NonNull<()>) -> Self {
        Self(ptr)
    }

    /// Get racy atomic pointer as a non-null pointer.
    ///
    /// # Soundness
    ///
    /// While this provides a Rust-recognisable pointer, this pointer is
    /// invalid and cannot be used to read or write any data. The only
    /// meaningful operation on the pointer is to perform possible offsetting.
    pub fn as_ptr(self) -> NonNull<()> {
        self.0
    }
}

/// An opaque pointer to a byte of memory implementing the ECMAScript atomic
/// memory model.
pub struct RacyAtomicU8(NonNull<()>, PhantomData<u8>);

impl RacyAtomicU8 {
    fn from_ptr(ptr: NonNull<()>) -> Self {
        Self(ptr, PhantomData)
    }

    fn as_ptr(&self) -> NonNull<()> {
        self.0
    }

    /// Stores a value into the racy atomic integer if the current value is the
    /// same as the `current` value.
    ///
    /// The return value is a result indicating whether the new value was
    /// written and containing the previous value. On success this value is
    /// guaranteed to be equal to `current`.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn compare_exchange(&self, current: u8, new: u8) -> Result<u8, u8> {
        let old = atomic_cmp_xchg_8_seq_cst(self.as_ptr(), current, new);
        if old == current { Ok(old) } else { Err(old) }
    }

    /// Adds to the current value, returning the previous value.
    ///
    /// This operation wraps around on overflow.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn fetch_add(&self, val: u8) -> u8 {
        atomic_add_8_seq_cst(self.as_ptr(), val)
    }

    /// Bitwise "and" with the current value.
    ///
    /// Performs a bitwise "and" operation on the current value and the argument `val`, and
    /// sets the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn fetch_and(&self, val: u8) -> u8 {
        atomic_and_8_seq_cst(self.as_ptr(), val)
    }

    /// Bitwise "or" with the current value.
    ///
    /// Performs a bitwise "or" operation on the current value and the argument `val`, and
    /// sets the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn fetch_or(&self, val: u8) -> u8 {
        atomic_or_8_seq_cst(self.as_ptr(), val)
    }

    /// Bitwise "xor" with the current value.
    ///
    /// Performs a bitwise "xor" operation on the current value and the argument `val`, and
    /// sets the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn fetch_xor(&self, val: u8) -> u8 {
        atomic_xor_8_seq_cst(self.as_ptr(), val)
    }

    /// Loads a value from the atomic integer.
    ///
    /// `load` takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. All ordering modes are possible.
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn load(&self, order: Ordering) -> u8 {
        if order == Ordering::SeqCst {
            atomic_load_8_seq_cst(self.as_ptr())
        } else {
            atomic_load_8_unsynchronized(self.as_ptr())
        }
    }

    /// Stores a value into the atomic integer.
    ///
    /// `store` takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. All ordering modes are possible.
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn store(&self, val: u8, order: Ordering) {
        if order == Ordering::SeqCst {
            atomic_store_8_seq_cst(self.as_ptr(), val)
        } else {
            atomic_store_8_unsynchronized(self.as_ptr(), val)
        }
    }

    /// Stores a value into the atomic integer, returning the previous value.
    ///
    /// `swap` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible.
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn swap(&self, val: u8) -> u8 {
        atomic_exchange_8_seq_cst(self.as_ptr(), val)
    }
}

pub struct RacyAtomicU16(NonNull<()>, PhantomData<u16>);

impl RacyAtomicU16 {
    fn from_ptr(ptr: NonNull<()>) -> Self {
        Self(ptr, PhantomData)
    }

    fn as_ptr(&self) -> NonNull<()> {
        self.0
    }

    /// Stores a value into the racy atomic integer if the current value is the
    /// same as the `current` value.
    ///
    /// The return value is a result indicating whether the new value was
    /// written and containing the previous value. On success this value is
    /// guaranteed to be equal to `current`.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn compare_exchange(&self, current: u16, new: u16) -> Result<u16, u16> {
        let old = atomic_cmp_xchg_16_seq_cst(self.as_ptr(), current, new);
        if old == current { Ok(old) } else { Err(old) }
    }

    /// Adds to the current value, returning the previous value.
    ///
    /// This operation wraps around on overflow.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn fetch_add(&self, val: u16) -> u16 {
        atomic_add_16_seq_cst(self.as_ptr(), val)
    }

    /// Bitwise "and" with the current value.
    ///
    /// Performs a bitwise "and" operation on the current value and the argument `val`, and
    /// sets the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn fetch_and(&self, val: u16) -> u16 {
        atomic_and_16_seq_cst(self.as_ptr(), val)
    }

    /// Bitwise "or" with the current value.
    ///
    /// Performs a bitwise "or" operation on the current value and the argument `val`, and
    /// sets the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn fetch_or(&self, val: u16) -> u16 {
        atomic_or_16_seq_cst(self.as_ptr(), val)
    }

    /// Bitwise "xor" with the current value.
    ///
    /// Performs a bitwise "xor" operation on the current value and the argument `val`, and
    /// sets the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn fetch_xor(&self, val: u16) -> u16 {
        atomic_xor_16_seq_cst(self.as_ptr(), val)
    }

    /// Loads a value from the atomic integer.
    ///
    /// `load` takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. All ordering modes are possible.
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn load(&self, order: Ordering) -> u16 {
        if order == Ordering::SeqCst {
            atomic_load_16_seq_cst(self.as_ptr())
        } else {
            atomic_load_16_unsynchronized(self.as_ptr())
        }
    }

    /// Stores a value into the atomic integer.
    ///
    /// `store` takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. All ordering modes are possible.
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn store(&self, val: u16, order: Ordering) {
        if order == Ordering::SeqCst {
            atomic_store_16_seq_cst(self.as_ptr(), val)
        } else {
            atomic_store_16_unsynchronized(self.as_ptr(), val)
        }
    }

    /// Stores a value into the atomic integer, returning the previous value.
    ///
    /// `swap` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible.
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn swap(&self, val: u16) -> u16 {
        atomic_exchange_16_seq_cst(self.as_ptr(), val)
    }
}

pub struct RacyAtomicU32(NonNull<()>, PhantomData<u32>);

impl RacyAtomicU32 {
    fn from_ptr(ptr: NonNull<()>) -> Self {
        Self(ptr, PhantomData)
    }

    fn as_ptr(&self) -> NonNull<()> {
        self.0
    }

    /// Stores a value into the racy atomic integer if the current value is the
    /// same as the `current` value.
    ///
    /// The return value is a result indicating whether the new value was
    /// written and containing the previous value. On success this value is
    /// guaranteed to be equal to `current`.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn compare_exchange(&self, current: u32, new: u32) -> Result<u32, u32> {
        let old = atomic_cmp_xchg_32_seq_cst(self.as_ptr(), current, new);
        if old == current { Ok(old) } else { Err(old) }
    }

    /// Adds to the current value, returning the previous value.
    ///
    /// This operation wraps around on overflow.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn fetch_add(&self, val: u32) -> u32 {
        atomic_add_32_seq_cst(self.as_ptr(), val)
    }

    /// Bitwise "and" with the current value.
    ///
    /// Performs a bitwise "and" operation on the current value and the argument `val`, and
    /// sets the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn fetch_and(&self, val: u32) -> u32 {
        atomic_and_32_seq_cst(self.as_ptr(), val)
    }

    /// Bitwise "or" with the current value.
    ///
    /// Performs a bitwise "or" operation on the current value and the argument `val`, and
    /// sets the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn fetch_or(&self, val: u32) -> u32 {
        atomic_or_32_seq_cst(self.as_ptr(), val)
    }

    /// Bitwise "xor" with the current value.
    ///
    /// Performs a bitwise "xor" operation on the current value and the argument `val`, and
    /// sets the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn fetch_xor(&self, val: u32) -> u32 {
        atomic_xor_32_seq_cst(self.as_ptr(), val)
    }

    /// Loads a value from the atomic integer.
    ///
    /// `load` takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. All ordering modes are possible.
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn load(&self, order: Ordering) -> u32 {
        if order == Ordering::SeqCst {
            atomic_load_32_seq_cst(self.as_ptr())
        } else {
            atomic_load_32_unsynchronized(self.as_ptr())
        }
    }

    /// Stores a value into the atomic integer.
    ///
    /// `store` takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. All ordering modes are possible.
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn store(&self, val: u32, order: Ordering) {
        if order == Ordering::SeqCst {
            atomic_store_32_seq_cst(self.as_ptr(), val)
        } else {
            atomic_store_32_unsynchronized(self.as_ptr(), val)
        }
    }

    /// Stores a value into the atomic integer, returning the previous value.
    ///
    /// `swap` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible.
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn swap(&self, val: u32) -> u32 {
        atomic_exchange_32_seq_cst(self.as_ptr(), val)
    }
}

pub struct RacyAtomicU64(NonNull<()>, PhantomData<u64>);

impl RacyAtomicU64 {
    fn from_ptr(ptr: NonNull<()>) -> Self {
        Self(ptr, PhantomData)
    }

    fn as_ptr(&self) -> NonNull<()> {
        self.0
    }

    /// Stores a value into the racy atomic integer if the current value is the
    /// same as the `current` value.
    ///
    /// The return value is a result indicating whether the new value was
    /// written and containing the previous value. On success this value is
    /// guaranteed to be equal to `current`.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm"
    ))]
    pub fn compare_exchange(&self, current: u64, new: u64) -> Result<u64, u64> {
        let old = atomic_cmp_xchg_64_seq_cst(self.as_ptr(), current, new);
        if old == current { Ok(old) } else { Err(old) }
    }

    /// Adds to the current value, returning the previous value.
    ///
    /// This operation wraps around on overflow.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
    pub fn fetch_add(&self, val: u64) -> u64 {
        atomic_add_64_seq_cst(self.as_ptr(), val)
    }

    /// Bitwise "and" with the current value.
    ///
    /// Performs a bitwise "and" operation on the current value and the argument `val`, and
    /// sets the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
    pub fn fetch_and(&self, val: u64) -> u64 {
        atomic_and_64_seq_cst(self.as_ptr(), val)
    }

    /// Bitwise "or" with the current value.
    ///
    /// Performs a bitwise "or" operation on the current value and the argument `val`, and
    /// sets the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
    pub fn fetch_or(&self, val: u64) -> u64 {
        atomic_or_64_seq_cst(self.as_ptr(), val)
    }

    /// Bitwise "xor" with the current value.
    ///
    /// Performs a bitwise "xor" operation on the current value and the argument `val`, and
    /// sets the new value to the result.
    ///
    /// Returns the previous value.
    ///
    /// The [`Ordering`] of the operation is always [`SeqCst`].
    ///
    /// [`Ordering`]: crate::Ordering
    /// [`SeqCst`]: crate::Ordering::SeqCst
    #[inline]
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
    pub fn fetch_xor(&self, val: u64) -> u64 {
        atomic_xor_64_seq_cst(self.as_ptr(), val)
    }

    /// Loads a value from the atomic integer.
    ///
    /// `load` takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. All ordering modes are possible.
    #[inline]
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
    pub fn load(&self, order: Ordering) -> u64 {
        if order == Ordering::SeqCst {
            atomic_load_64_seq_cst(self.as_ptr())
        } else {
            atomic_load_64_unsynchronized(self.as_ptr())
        }
    }

    /// Stores a value into the atomic integer.
    ///
    /// `store` takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. All ordering modes are possible.
    #[inline]
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
    pub fn store(&self, val: u64, order: Ordering) {
        if order == Ordering::SeqCst {
            atomic_store_64_seq_cst(self.as_ptr(), val)
        } else {
            atomic_store_64_unsynchronized(self.as_ptr(), val)
        }
    }

    /// Stores a value into the atomic integer, returning the previous value.
    ///
    /// `swap` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible.
    #[inline]
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
    pub fn swap(&self, val: u64) -> u64 {
        atomic_exchange_64_seq_cst(self.as_ptr(), val)
    }
}

fn can_copy_aligned<const ALIGNMENT: usize>(
    src: NonNull<()>,
    dst: NonNull<()>,
    lim: NonNull<()>,
) -> bool {
    const { assert!(ALIGNMENT.is_power_of_two() && ALIGNMENT > 0) };
    ((src.addr().get() | dst.addr().get() | lim.addr().get()) & (ALIGNMENT - 1)) == 0
}

fn can_align_to<const ALIGNMENT: usize>(src: NonNull<()>, dst: NonNull<()>) -> bool {
    const { assert!(ALIGNMENT.is_power_of_two() && ALIGNMENT > 0) };
    ((src.addr().get() ^ dst.addr().get()) & (ALIGNMENT - 1)) == 0
}

const BLOCK_MASK: usize = BLOCK_SIZE - 1;
const WORD_MASK: usize = WORD_SIZE - 1;

unsafe fn unordered_memcpy_down_unsynchronized(
    mut src: NonNull<()>,
    mut dst: NonNull<()>,
    count: usize,
) {
    // SAFETY: Caller guaranteed.
    let lim = unsafe { src.byte_add(count) };

    // Set up bulk copying.  The cases are ordered the way they are on the
    // assumption that if we can achieve aligned copies even with a little
    // preprocessing then that is better than unaligned copying on a platform
    // that supports it.

    if count >= WORD_SIZE {
        let copy_block: fn(NonNull<()>, NonNull<()>);
        let copy_word: fn(NonNull<()>, NonNull<()>);

        if can_align_to::<WORD_SIZE>(src, dst) {
            // SAFETY: Cannot overflow because count >= WORD_SIZE.
            let cutoff = unsafe { src.byte_add(src.cast::<u8>().align_offset(WORD_SIZE)) };
            debug_assert!(cutoff <= lim); // because count >= WORD_SIZE

            // Copy initial bytes to align to word size.
            (src, dst) = atomic_copy_down_no_tear_if_aligned_unsynchronized(src, dst, cutoff);

            copy_block = atomic_copy_block_down_unsynchronized;
            copy_word = atomic_copy_word_unsynchronized;
        } else if UNALIGNED_ACCESS_IS_OK {
            copy_block = atomic_copy_block_down_unsynchronized;
            copy_word = atomic_copy_word_unsynchronized;
        } else {
            copy_block = atomic_copy_unaligned_block_down_unsynchronized;
            copy_word = atomic_copy_unaligned_word_down_unsynchronized;
        }

        // Bulk copy, first larger blocks and then individual words.

        let block_lim = unsafe { src.byte_add(lim.byte_offset_from_unsigned(src) & !BLOCK_MASK) };
        while src < block_lim {
            copy_block(src, dst);
            // SAFETY: Checked.
            unsafe {
                dst = dst.byte_add(BLOCK_SIZE);
                src = src.byte_add(BLOCK_SIZE);
            }
        }

        let word_lim = unsafe { src.byte_add(lim.byte_offset_from_unsigned(src) & !WORD_MASK) };
        while src < word_lim {
            copy_word(src, dst);
            // SAFETY: Checked.
            unsafe {
                dst = dst.byte_add(WORD_SIZE);
                src = src.byte_add(WORD_SIZE);
            }
        }
    }

    // Copy any remaining tail.

    atomic_copy_down_no_tear_if_aligned_unsynchronized(src, dst, lim);
}

unsafe fn unordered_memcpy_up_unsynchronized(
    mut src: NonNull<()>,
    mut dst: NonNull<()>,
    count: usize,
) {
    let lim = src;

    // SAFETY: Caller guaranteed.
    unsafe {
        src = src.byte_add(count);
        dst = dst.byte_add(count);
    }

    // Set up bulk copying.  The cases are ordered the way they are on the
    // assumption that if we can achieve aligned copies even with a little
    // preprocessing then that is better than unaligned copying on a platform
    // that supports it.

    if count >= WORD_SIZE {
        let copy_block: fn(NonNull<()>, NonNull<()>);
        let copy_word: fn(NonNull<()>, NonNull<()>);

        if can_align_to::<WORD_SIZE>(src, dst) {
            // SAFETY: src necessary has non-zero bits beyond the WORD_MASK
            // bottom bits as `count >= WORD_SIZE` and
            // `src = src.byte_add(count)`.
            let cutoff_addr = unsafe { NonZeroUsize::new_unchecked(src.addr().get() & !WORD_MASK) };
            let cutoff = src.with_addr(cutoff_addr);
            debug_assert!(cutoff >= lim); // Because count >= WORD_SIZE

            // Copy initial bytes to align to word size.
            (src, dst) = atomic_copy_up_no_tear_if_aligned_unsynchronized(src, dst, cutoff);

            copy_block = atomic_copy_block_up_unsynchronized;
            copy_word = atomic_copy_word_unsynchronized;
        } else if UNALIGNED_ACCESS_IS_OK {
            copy_block = atomic_copy_block_up_unsynchronized;
            copy_word = atomic_copy_word_unsynchronized;
        } else {
            copy_block = atomic_copy_unaligned_block_up_unsynchronized;
            copy_word = atomic_copy_unaligned_word_up_unsynchronized;
        }

        // Bulk copy, first larger blocks and then individual words.

        let block_lim = unsafe { src.byte_sub(src.byte_offset_from_unsigned(lim) & !BLOCK_MASK) };
        while src > block_lim {
            // SAFETY: Checked.
            unsafe {
                src = src.byte_sub(BLOCK_SIZE);
                dst = dst.byte_sub(BLOCK_SIZE);
            }
            copy_block(src, dst);
        }

        let word_lim = unsafe { src.byte_sub(src.byte_offset_from_unsigned(lim) & !WORD_MASK) };
        while src > word_lim {
            // SAFETY: Checked.
            unsafe {
                src = src.byte_sub(WORD_SIZE);
                dst = dst.byte_sub(WORD_SIZE);
            }
            copy_word(src, dst);
        }
    }

    // Copy any remaining tail.

    atomic_copy_up_no_tear_if_aligned_unsynchronized(src, dst, lim);
}

/// Copies `count` bytes from `src` to `dst`. The source
/// and destination must *not* overlap.
///
/// For regions of memory which might overlap, use [`unordered_copy`] instead.
///
/// `unordered_copy_nonoverlapping` is semantically equivalent to C's
/// [`memcpy`], but with the source and destination arguments swapped, and with
/// data races allowed.
///
/// [`memcpy`]: https://en.cppreference.com/w/c/string/byte/memcpy
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `src` must be [valid] for reads of `count` bytes.
///
/// * `dst` must be [valid] for writes of `count` bytes.
///
/// * Both `src` and `dst` must be properly aligned.
///
/// * The region of memory beginning at `src` with a size of `count` bytes must
///   *not* overlap with the region of memory beginning at `dst` with the same
///   size.
///
/// [valid]: https://doc.rust-lang.org/stable/core/ptr/#safety
pub unsafe fn unordered_copy_nonoverlapping(src: RacyAtomicU8, dst: RacyAtomicU8, count: usize) {
    unsafe { unordered_memcpy_down_unsynchronized(src.as_ptr(), dst.as_ptr(), count) };
}

/// Copies `count` bytes from `src` to `dst`. The source and destination may
/// overlap.
///
/// If the source and destination will *never* overlap,
/// [`unordered_copy_nonoverlapping`] can be used instead.
///
/// `unordered_copy` is semantically equivalent to C's [`memmove`], but with
/// the source and destination arguments swapped, and with data races allowed.
/// Copying takes place as if the bytes were copied from `src` to a temporary
/// array and then copied from the array to `dst`.
///
/// [`memmove`]: https://en.cppreference.com/w/c/string/byte/memmove
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `src` must be [valid] for reads of `count` bytes.
///
/// * `dst` must be [valid] for writes of `count` bytes, and must remain valid even
///   when `src` is read for `count` bytes. (This means if the memory ranges
///   overlap, the `dst` pointer must not be invalidated by `src` reads.)
///
/// [valid]: https://doc.rust-lang.org/stable/core/ptr/#safety
pub unsafe fn unordered_copy(src: RacyAtomicU8, dst: RacyAtomicU8, count: usize) {
    if dst.as_ptr() <= src.as_ptr() {
        unsafe { unordered_memcpy_down_unsynchronized(src.as_ptr(), dst.as_ptr(), count) };
    } else {
        unsafe { unordered_memcpy_up_unsynchronized(src.as_ptr(), dst.as_ptr(), count) };
    }
}

/// Copy a datum smaller than `WORD_SIZE`. Prevents tearing when `src` and `dst`
/// are both aligned.
///
/// No tearing is a requirement for integer TypedArrays.
///
/// https://tc39.es/ecma262/#sec-isnotearconfiguration
/// https://tc39.es/ecma262/#sec-tear-free-aligned-reads
/// https://tc39.es/ecma262/#sec-valid-executions
#[inline(always)]
fn atomic_copy_up_no_tear_if_aligned_unsynchronized(
    mut src: NonNull<()>,
    mut dst: NonNull<()>,
    src_begin: NonNull<()>,
) -> (NonNull<()>, NonNull<()>) {
    // SAFETY: Checking caller guarantees.
    unsafe {
        assert_unchecked(src >= src_begin);
        assert_unchecked(src.byte_offset_from_unsigned(src_begin) < WORD_SIZE);
    }

    if WORD_SIZE > 4 && can_copy_aligned::<4>(src, dst, src_begin) {
        const { assert!(WORD_SIZE <= 8, "copies 32-bits at most once") };

        if src > src_begin {
            const SIZE: usize = size_of::<u32>();
            // SAFETY: checked.
            unsafe {
                src = src.byte_sub(SIZE);
                dst = dst.byte_sub(SIZE);
            }
            atomic_copy32_unsynchronized(src, dst);
        }
    } else if can_copy_aligned::<2>(src, dst, src_begin) {
        while src > src_begin {
            const SIZE: usize = size_of::<u16>();
            // SAFETY: checked.
            unsafe {
                src = src.byte_sub(SIZE);
                dst = dst.byte_sub(SIZE);
            }
            atomic_copy16_unsynchronized(src, dst);
        }
    } else {
        while src > src_begin {
            const SIZE: usize = size_of::<u8>();
            // SAFETY: checked.
            unsafe {
                src = src.byte_sub(SIZE);
                dst = dst.byte_sub(SIZE);
            }
            atomic_copy8_unsynchronized(src, dst);
        }
    }
    (src, dst)
}

#[inline(always)]
fn atomic_copy_down_no_tear_if_aligned_unsynchronized(
    mut src: NonNull<()>,
    mut dst: NonNull<()>,
    src_end: NonNull<()>,
) -> (NonNull<()>, NonNull<()>) {
    // SAFETY: Checked by caller.
    unsafe { assert_unchecked(src <= src_end) };
    // SAFETY: Checked by caller.
    unsafe { assert_unchecked(src_end.byte_offset_from_unsigned(src) < WORD_SIZE) };

    if WORD_SIZE > 4 && can_copy_aligned::<4>(src, dst, src_end) {
        const { assert!(WORD_SIZE <= 8, "copies 32-bits at most once") };
        if src < src_end {
            atomic_copy32_unsynchronized(src, dst);
            const SIZE: usize = size_of::<u32>();
            // SAFETY: checked.
            unsafe {
                src = src.byte_add(SIZE);
                dst = dst.byte_add(SIZE);
            }
        }
    } else if can_copy_aligned::<2>(src, dst, src_end) {
        while src < src_end {
            atomic_copy16_unsynchronized(src, dst);
            const SIZE: usize = size_of::<u16>();
            // SAFETY: checked.
            unsafe {
                src = src.byte_add(SIZE);
                dst = dst.byte_add(SIZE);
            }
        }
    } else {
        while src < src_end {
            atomic_copy8_unsynchronized(src, dst);
            const SIZE: usize = size_of::<u8>();
            // SAFETY: checked.
            unsafe {
                src = src.byte_add(SIZE);
                dst = dst.byte_add(SIZE);
            }
        }
    }
    (src, dst)
}

#[cfg(test)]
mod test {
    use std::ptr::NonNull;

    use crate::*;

    #[test]
    fn test_load() {
        let dst = NonNull::from(Box::leak(Box::new([u64::MAX; 1]))).cast::<()>();

        assert_eq!(atomic_load_8_unsynchronized(dst), u8::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX);

        assert_eq!(atomic_load_16_unsynchronized(dst), u16::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX);

        assert_eq!(atomic_load_32_unsynchronized(dst), u32::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX);

        assert_eq!(atomic_load_64_unsynchronized(dst), u64::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX);

        assert_eq!(atomic_load_8_seq_cst(dst), u8::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX);

        assert_eq!(atomic_load_16_seq_cst(dst), u16::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX);

        assert_eq!(atomic_load_32_seq_cst(dst), u32::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX);

        assert_eq!(atomic_load_64_seq_cst(dst), u64::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX);

        let _ = unsafe { Box::from_raw(dst.cast::<u64>().as_ptr()) };
    }

    #[test]
    fn test_store() {
        let dst = NonNull::from(Box::leak(Box::new([0u64; 1]))).cast::<()>();

        atomic_store_8_unsynchronized(dst, u8::MAX);
        assert_eq!(atomic_load_8_unsynchronized(dst), u8::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u8::MAX as u64);

        atomic_store_16_unsynchronized(dst, u16::MAX);
        assert_eq!(atomic_load_16_unsynchronized(dst), u16::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u16::MAX as u64);

        atomic_store_32_unsynchronized(dst, u32::MAX);
        assert_eq!(atomic_load_32_unsynchronized(dst), u32::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u32::MAX as u64);

        atomic_store_64_unsynchronized(dst, u64::MAX);
        assert_eq!(atomic_load_64_unsynchronized(dst), u64::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX);

        atomic_store_64_unsynchronized(dst, 0x0);
        assert_eq!(atomic_load_64_unsynchronized(dst), 0x0);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0x0);

        atomic_store_8_seq_cst(dst, u8::MAX);
        assert_eq!(atomic_load_8_seq_cst(dst), u8::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u8::MAX as u64);

        atomic_store_16_seq_cst(dst, u16::MAX);
        assert_eq!(atomic_load_16_seq_cst(dst), u16::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u16::MAX as u64);

        atomic_store_32_seq_cst(dst, u32::MAX);
        assert_eq!(atomic_load_32_seq_cst(dst), u32::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u32::MAX as u64);

        atomic_store_64_seq_cst(dst, u64::MAX);
        assert_eq!(atomic_load_64_seq_cst(dst), u64::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX);

        let _ = unsafe { Box::from_raw(dst.cast::<u64>().as_ptr()) };
    }

    #[test]
    fn test_exchange() {
        let dst = NonNull::from(Box::leak(Box::new([0u64; 1]))).cast::<()>();

        assert_eq!(atomic_exchange_8_seq_cst(dst, u8::MAX), 0, "u8 initial");
        assert_eq!(atomic_exchange_8_seq_cst(dst, 0), u8::MAX, "u8 subsequent");
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);

        assert_eq!(atomic_exchange_16_seq_cst(dst, u16::MAX), 0, "u16 initial");
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u16::MAX as u64);
        assert_eq!(
            atomic_exchange_16_seq_cst(dst, 0),
            u16::MAX,
            "u16 subsequent"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);

        assert_eq!(atomic_exchange_32_seq_cst(dst, u32::MAX), 0, "u32 initial");
        assert_eq!(
            atomic_exchange_32_seq_cst(dst, 0),
            u32::MAX,
            "u32 subsequent"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);

        assert_eq!(atomic_exchange_64_seq_cst(dst, u64::MAX), 0, "u64 initial");
        assert_eq!(
            atomic_exchange_64_seq_cst(dst, 0),
            u64::MAX,
            "u64 subsequent"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);

        let _ = unsafe { Box::from_raw(dst.cast::<u64>().as_ptr()) };
    }

    #[test]
    fn test_compare_exchange() {
        let dst = NonNull::from(Box::leak(Box::new([0u64; 1]))).cast::<()>();

        assert_eq!(
            atomic_cmp_xchg_8_seq_cst(dst, u8::MAX, u8::MAX),
            0,
            "u8 initial"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);
        assert_eq!(atomic_cmp_xchg_8_seq_cst(dst, 0, u8::MAX), 0, "u8 initial");
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u8::MAX as u64);
        assert_eq!(
            atomic_cmp_xchg_8_seq_cst(dst, 0, 0),
            u8::MAX,
            "u8 subsequent"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u8::MAX as u64);
        assert_eq!(
            atomic_cmp_xchg_8_seq_cst(dst, u8::MAX, 0),
            u8::MAX,
            "u8 subsequent"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);

        assert_eq!(
            atomic_cmp_xchg_16_seq_cst(dst, u16::MAX, u16::MAX),
            0,
            "u16 initial"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);
        assert_eq!(
            atomic_cmp_xchg_16_seq_cst(dst, 0, u16::MAX),
            0,
            "u16 initial"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u16::MAX as u64);
        assert_eq!(
            atomic_cmp_xchg_16_seq_cst(dst, 0, 0),
            u16::MAX,
            "u16 subsequent"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u16::MAX as u64);
        assert_eq!(
            atomic_cmp_xchg_16_seq_cst(dst, u16::MAX, 0),
            u16::MAX,
            "u16 subsequent"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);

        assert_eq!(
            atomic_cmp_xchg_32_seq_cst(dst, u32::MAX, u32::MAX),
            0,
            "u32 initial"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);
        assert_eq!(
            atomic_cmp_xchg_32_seq_cst(dst, 0, u32::MAX),
            0,
            "u32 initial"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u32::MAX as u64);
        assert_eq!(
            atomic_cmp_xchg_32_seq_cst(dst, 0, 0),
            u32::MAX,
            "u32 subsequent"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u32::MAX as u64);
        assert_eq!(
            atomic_cmp_xchg_32_seq_cst(dst, u32::MAX, 0),
            u32::MAX,
            "u32 subsequent"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);

        assert_eq!(
            atomic_cmp_xchg_64_seq_cst(dst, u64::MAX, u64::MAX),
            0,
            "u64 initial"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);
        assert_eq!(
            atomic_cmp_xchg_64_seq_cst(dst, 0, u64::MAX),
            0,
            "u64 initial"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX);
        assert_eq!(
            atomic_cmp_xchg_64_seq_cst(dst, 0, 0),
            u64::MAX,
            "u64 subsequent"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX);
        assert_eq!(
            atomic_cmp_xchg_64_seq_cst(dst, u64::MAX, 0),
            u64::MAX,
            "u64 subsequent"
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);

        let _ = unsafe { Box::from_raw(dst.cast::<u64>().as_ptr()) };
    }

    #[test]
    fn test_add() {
        let dst = NonNull::from(Box::leak(Box::new([0u64; 1]))).cast::<()>();

        assert_eq!(atomic_add_8_seq_cst(dst, u8::MAX), 0);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u8::MAX as u64);
        assert_eq!(atomic_add_8_seq_cst(dst, 1), u8::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);

        assert_eq!(atomic_add_16_seq_cst(dst, u16::MAX), 0);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u16::MAX as u64);
        assert_eq!(atomic_add_16_seq_cst(dst, 1), u16::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);

        assert_eq!(atomic_add_32_seq_cst(dst, u32::MAX), 0);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u32::MAX as u64);
        assert_eq!(atomic_add_32_seq_cst(dst, 1), u32::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);

        assert_eq!(atomic_add_64_seq_cst(dst, u64::MAX), 0);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX);
        assert_eq!(atomic_add_64_seq_cst(dst, 1), u64::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);

        let _ = unsafe { Box::from_raw(dst.cast::<u64>().as_ptr()) };
    }

    #[test]
    fn test_and() {
        let dst = NonNull::from(Box::leak(Box::new([0u64; 1]))).cast::<()>();

        assert_eq!(atomic_and_8_seq_cst(dst, u8::MAX), 0);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);
        atomic_store_64_unsynchronized(dst, u64::MAX);
        assert_eq!(atomic_and_8_seq_cst(dst, u8::MAX), u8::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX);
        assert_eq!(atomic_and_8_seq_cst(dst, 0xF0), u8::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX - 0xF);
        assert_eq!(atomic_and_8_seq_cst(dst, 0), 0xF0);
        assert_eq!(
            unsafe { dst.cast::<u64>().read() },
            u64::MAX - u8::MAX as u64
        );
        unsafe {
            dst.cast::<u64>().write(0);
        }

        assert_eq!(atomic_and_16_seq_cst(dst, u16::MAX), 0);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);
        atomic_store_64_unsynchronized(dst, u64::MAX);
        assert_eq!(atomic_and_16_seq_cst(dst, u16::MAX), u16::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX);
        assert_eq!(atomic_and_16_seq_cst(dst, 0xFF00), u16::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX - 0xFF);
        assert_eq!(atomic_and_16_seq_cst(dst, 0), 0xFF00);
        assert_eq!(
            unsafe { dst.cast::<u64>().read() },
            u64::MAX - u16::MAX as u64
        );
        unsafe {
            dst.cast::<u64>().write(0);
        }

        assert_eq!(atomic_and_32_seq_cst(dst, u32::MAX), 0);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);
        atomic_store_64_unsynchronized(dst, u64::MAX);
        assert_eq!(atomic_and_32_seq_cst(dst, u32::MAX), u32::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX);
        assert_eq!(atomic_and_32_seq_cst(dst, 0xFFFF_0000), u32::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX - 0xFFFF);
        assert_eq!(atomic_and_32_seq_cst(dst, 0), 0xFFFF_0000);
        assert_eq!(
            unsafe { dst.cast::<u64>().read() },
            u64::MAX - u32::MAX as u64
        );
        unsafe {
            dst.cast::<u64>().write(0);
        }

        assert_eq!(atomic_and_64_seq_cst(dst, u64::MAX), 0);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);
        atomic_store_64_unsynchronized(dst, u64::MAX);
        assert_eq!(atomic_and_64_seq_cst(dst, u64::MAX), u64::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, u64::MAX);
        assert_eq!(atomic_and_64_seq_cst(dst, 0xFFFF_0000_FFFF_0000), u64::MAX);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xFFFF_0000_FFFF_0000);
        assert_eq!(
            atomic_and_64_seq_cst(dst, 0x0_FFFF_0000_FFFF),
            0xFFFF_0000_FFFF_0000
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0);

        let _ = unsafe { Box::from_raw(dst.cast::<u64>().as_ptr()) };
    }

    #[test]
    fn test_or() {
        let dst = NonNull::from(Box::leak(Box::new([0u64; 1]))).cast::<()>();

        assert_eq!(atomic_or_8_seq_cst(dst, 0x73), 0);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0x73);
        assert_eq!(atomic_or_8_seq_cst(dst, 0x1B), 0x73);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0x7B);
        assert_eq!(atomic_or_8_seq_cst(dst, 0xF0), 0x7B);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xFB);
        assert_eq!(atomic_or_8_seq_cst(dst, 0x00), 0xFB);
        assert_eq!(atomic_or_8_seq_cst(dst, 0xFF), 0xFB);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xFF);
        unsafe {
            dst.cast::<u64>().write(0);
        }

        assert_eq!(atomic_or_16_seq_cst(dst, 0xB182), 0);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xB182);
        assert_eq!(atomic_or_16_seq_cst(dst, 0x02C3), 0xB182);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xB3C3);
        assert_eq!(atomic_or_16_seq_cst(dst, 0xFF00), 0xB3C3);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xFFC3);
        assert_eq!(atomic_or_16_seq_cst(dst, 0), 0xFFC3);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xFFC3);
        assert_eq!(atomic_or_16_seq_cst(dst, 0x00FF), 0xFFC3);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xFFFF);
        assert_eq!(atomic_or_16_seq_cst(dst, 0), 0xFFFF);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xFFFF);
        unsafe {
            dst.cast::<u64>().write(0);
        }

        assert_eq!(atomic_or_32_seq_cst(dst, 0x01A4_1005), 0);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0x01A4_1005);
        assert_eq!(atomic_or_32_seq_cst(dst, 0x5502_D581), 0x01A4_1005);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0x55A6_D585);
        assert_eq!(atomic_or_32_seq_cst(dst, 0xFF00_FF00), 0x55A6_D585);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xFFA6_FF85);
        assert_eq!(atomic_or_32_seq_cst(dst, 0), 0xFFA6_FF85);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xFFA6_FF85);
        assert_eq!(atomic_or_32_seq_cst(dst, 0x00FF_00FF), 0xFFA6_FF85);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xFFFF_FFFF);
        unsafe {
            dst.cast::<u64>().write(0);
        }

        assert_eq!(atomic_or_64_seq_cst(dst, 0xABCD_3456_01A4_1005), 0);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xABCD_3456_01A4_1005);
        assert_eq!(
            atomic_or_64_seq_cst(dst, 0x0F25_0021_232B_C34A),
            0xABCD_3456_01A4_1005
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xAFED_3477_23AF_D34F);
        assert_eq!(
            atomic_or_64_seq_cst(dst, 0xFF00_FF00_FF00_FF00),
            0xAFED_3477_23AF_D34F
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xFFED_FF77_FFAF_FF4F);
        assert_eq!(atomic_or_64_seq_cst(dst, 0), 0xFFED_FF77_FFAF_FF4F);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xFFED_FF77_FFAF_FF4F);
        assert_eq!(
            atomic_or_64_seq_cst(dst, 0x00FF_00FF_00FF_00FF),
            0xFFED_FF77_FFAF_FF4F
        );
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xFFFF_FFFF_FFFF_FFFF);

        let _ = unsafe { Box::from_raw(dst.cast::<u64>().as_ptr()) };
    }

    #[test]
    fn test_xor() {
        let foo = NonNull::from(Box::leak(Box::new([0u64; 1]))).cast::<()>();

        assert_eq!(atomic_xor_8_seq_cst(foo, 0x73), 0);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0x73);
        assert_eq!(atomic_xor_8_seq_cst(foo, 0x1B), 0x73);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0x68);
        assert_eq!(atomic_xor_8_seq_cst(foo, 0xF0), 0x68);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0x98);
        assert_eq!(atomic_xor_8_seq_cst(foo, 0x00), 0x98);
        assert_eq!(atomic_xor_8_seq_cst(foo, 0xFF), 0x98);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0x67);
        assert_eq!(atomic_xor_8_seq_cst(foo, 0x67), 0x67);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0);

        assert_eq!(atomic_xor_16_seq_cst(foo, 0xB182), 0);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0xB182);
        assert_eq!(atomic_xor_16_seq_cst(foo, 0x02C3), 0xB182);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0xB341);
        assert_eq!(atomic_xor_16_seq_cst(foo, 0xFF00), 0xB341);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0x4C41);
        assert_eq!(atomic_xor_16_seq_cst(foo, 0), 0x4C41);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0x4C41);
        assert_eq!(atomic_xor_16_seq_cst(foo, 0x00FF), 0x4C41);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0x4CBE);
        assert_eq!(atomic_xor_16_seq_cst(foo, 0xFFFF), 0x4CBE);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0xB341);
        assert_eq!(atomic_xor_16_seq_cst(foo, 0xB341), 0xB341);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0);

        assert_eq!(atomic_xor_32_seq_cst(foo, 0xA34B_B182), 0);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0xA34B_B182);
        assert_eq!(atomic_xor_32_seq_cst(foo, 0x86D0_02C3), 0xA34B_B182);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0x259B_B341);
        assert_eq!(atomic_xor_32_seq_cst(foo, 0xFF00_FF00), 0x259B_B341);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0xDA9B_4C41);
        assert_eq!(atomic_xor_32_seq_cst(foo, 0), 0xDA9B_4C41);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0xDA9B_4C41);
        assert_eq!(atomic_xor_32_seq_cst(foo, 0x00FF_00FF), 0xDA9B_4C41);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0xDA64_4CBE);
        assert_eq!(atomic_xor_32_seq_cst(foo, 0xFFFF_FFFF), 0xDA64_4CBE);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0x259B_B341);
        assert_eq!(atomic_xor_32_seq_cst(foo, 0x259B_B341), 0x259B_B341);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0);

        assert_eq!(atomic_xor_64_seq_cst(foo, 0x0567_98E0_A34B_B182), 0);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0x0567_98E0_A34B_B182);
        assert_eq!(
            atomic_xor_64_seq_cst(foo, 0x1135_C732_86D0_02C3),
            0x0567_98E0_A34B_B182
        );
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0x1452_5FD2_259B_B341);
        assert_eq!(
            atomic_xor_64_seq_cst(foo, 0xFF00_FF00_FF00_FF00),
            0x1452_5FD2_259B_B341
        );
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0xEB52_A0D2_DA9B_4C41);
        assert_eq!(atomic_xor_64_seq_cst(foo, 0), 0xEB52_A0D2_DA9B_4C41);
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0xEB52_A0D2_DA9B_4C41);
        assert_eq!(
            atomic_xor_64_seq_cst(foo, 0x00FF_00FF_00FF_00FF),
            0xEB52_A0D2_DA9B_4C41
        );
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0xEBAD_A02D_DA64_4CBE);
        assert_eq!(
            atomic_xor_64_seq_cst(foo, 0xFFFF_FFFF_FFFF_FFFF),
            0xEBAD_A02D_DA64_4CBE
        );
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0x1452_5FD2_259B_B341);
        assert_eq!(
            atomic_xor_64_seq_cst(foo, 0x1452_5FD2_259B_B341),
            0x1452_5FD2_259B_B341
        );
        assert_eq!(unsafe { foo.cast::<u64>().read() }, 0);

        let _ = unsafe { Box::from_raw(foo.cast::<u64>().as_ptr()) };
    }

    #[test]
    fn test_copy() {
        let src = NonNull::from(Box::leak(Box::new([0u64; 16]))).cast::<()>();
        let dst = NonNull::from(Box::leak(Box::new([0u64; 16]))).cast::<()>();

        unsafe { src.cast::<u64>().write(0xFFFF_FFFF_FFFF_FFFF) };
        atomic_copy8_unsynchronized(src, dst);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xFF);
        atomic_copy16_unsynchronized(src, dst);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xFFFF);
        atomic_copy32_unsynchronized(src, dst);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xFFFF_FFFF);
        atomic_copy_word_unsynchronized(src, dst);
        assert_eq!(unsafe { dst.cast::<u64>().read() }, 0xFFFF_FFFF_FFFF_FFFF);

        unsafe { src.cast::<[u64; 16]>().as_mut().fill(0xF0F0_F0F0_F0F0_F0F0) };
        atomic_copy_block_down_unsynchronized(src, dst);
        assert_eq!(
            unsafe { src.cast::<[u8; size_of::<usize>() * 8]>().read() },
            unsafe { dst.cast::<[u8; size_of::<usize>() * 8]>().read() },
        );
        unsafe { src.cast::<[u64; 16]>().as_mut().fill(0x0F0F_0F0F_0F0F_0F0F) };
        atomic_copy_block_up_unsynchronized(src, dst);
        assert_eq!(
            unsafe { src.cast::<[u8; size_of::<usize>() * 8]>().read() },
            unsafe { dst.cast::<[u8; size_of::<usize>() * 8]>().read() },
        );
        unsafe { src.cast::<[u64; 16]>().as_mut().fill(0xABCD_EF01_2345_6789) };
        atomic_copy_unaligned_block_down_unsynchronized(unsafe { src.byte_add(1) }, dst);
        assert_eq!(
            unsafe {
                src.byte_add(1)
                    .cast::<[u8; size_of::<usize>() * 8]>()
                    .read()
            },
            unsafe { dst.cast::<[u8; size_of::<usize>() * 8]>().read() },
        );
        unsafe { src.cast::<[u64; 16]>().as_mut().fill(0xBCDE_F012_3456_789A) };
        atomic_copy_unaligned_block_up_unsynchronized(unsafe { src.byte_add(3) }, dst);
        assert_eq!(
            unsafe {
                src.byte_add(3)
                    .cast::<[u8; size_of::<usize>() * 8]>()
                    .read()
            },
            unsafe { dst.cast::<[u8; size_of::<usize>() * 8]>().read() },
        );
        unsafe { src.cast::<[u64; 16]>().as_mut().fill(0xCDEF_0123_4567_89AB) };
        unsafe { dst.cast::<[u64; 16]>().as_mut().fill(0) };
        atomic_copy_unaligned_word_up_unsynchronized(unsafe { src.byte_add(5) }, dst);
        assert_eq!(
            unsafe { src.byte_add(5).cast::<[u8; size_of::<usize>()]>().read() },
            unsafe { dst.cast::<[u8; size_of::<usize>()]>().read() },
        );
        unsafe { src.cast::<[u64; 16]>().as_mut().fill(0xDEF0_1234_5678_9ABC) };
        atomic_copy_unaligned_word_down_unsynchronized(unsafe { src.byte_add(7) }, dst);
        assert_eq!(
            unsafe { src.byte_add(7).cast::<[u8; size_of::<usize>()]>().read() },
            unsafe { dst.cast::<[u8; size_of::<usize>()]>().read() },
        );

        let _ = unsafe { Box::from_raw(src.cast::<u64>().as_ptr()) };
        let _ = unsafe { Box::from_raw(dst.cast::<u64>().as_ptr()) };
    }
}
