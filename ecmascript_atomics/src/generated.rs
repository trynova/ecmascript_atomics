// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Copyright of the originating code is owned by Firefox authors and Mozilla,
// modifications by Aapo Alasuutari.

//! See the comment in [./lib.rs] for details.

use core::ptr::NonNull;

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
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "mov {val}, [{ptr}]",
                fence!(false, x86),
                ptr = in(reg) $ptr.as_ptr(),
                val = lateout(reg_byte) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                "ldrb {val:w}, [{ptr}]",
                fence!($barrier, aarch64),
                ptr = in(reg) $ptr.as_ptr(),
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            core::arch::asm!(
                "ldrb {val:w}, [{ptr}]",
                fence!($barrier, arm),
                ptr = in(reg) $ptr.as_ptr(),
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        return z;
    };
    (u16, $ptr: ident, $barrier: tt) => {
        let z: u16;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "mov {val:x}, [{ptr}]",
                fence!(false, x86),
                ptr = in(reg) $ptr.as_ptr(),
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                "ldrh {val:w}, [{ptr}]",
                fence!($barrier, aarch64),
                ptr = in(reg) $ptr.as_ptr(),
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            core::arch::asm!(
                "ldrh {val:w}, [{ptr}]",
                fence!($barrier, arm),
                ptr = in(reg) $ptr.as_ptr(),
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        return z;
    };
    (u32, $ptr: ident, $barrier: tt) => {
        let z: u32;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "mov {val:e}, [{ptr}]",
                fence!(false, x86),
                ptr = in(reg) $ptr.as_ptr(),
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                "ldr {val:w}, [{ptr}]",
                fence!($barrier, aarch64),
                ptr = in(reg) $ptr.as_ptr(),
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        #[cfg(target_arch = "arm")]
        unsafe {
            core::arch::asm!(
                "ldr {val:w}, [{ptr}]",
                fence!($barrier, arm),
                ptr = in(reg) $ptr.as_ptr(),
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        return z;
    };
    (u64, $ptr: ident, $barrier: tt) => {
        let z: u64;
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::asm!(
                "mov {val:r}, [{ptr}]",
                fence!(false, x86),
                ptr = in(reg) $ptr.as_ptr(),
                val = lateout(reg) z,
                options(preserves_flags, nostack, pure, readonly)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                "ldr {val:x}, [{ptr}]",
                fence!($barrier, aarch64),
                ptr = in(reg) $ptr.as_ptr(),
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
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "mov [{ptr}], {val}",
                fence!($barrier, x86),
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
                val = in(reg) $val,
                options(preserves_flags, nostack)
            );
        }
    };
    (u16, $ptr: ident, $val: ident, $barrier: tt) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "mov [{ptr}], {val:x}",
                fence!($barrier, x86),
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
                val = in(reg) $val,
                options(preserves_flags, nostack)
            );
        }
    };
    (u32, $ptr: ident, $val: ident, $barrier: tt) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "mov [{ptr}], {val:e}",
                fence!($barrier, x86),
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
                val = in(reg) $val,
                options(preserves_flags, nostack)
            );
        }
    };
    (u64, $ptr: ident, $val: ident, $barrier: tt) => {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::asm!(
                "mov [{ptr}], {val:r}",
                fence!($barrier, x86),
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "xchg [{ptr}], {val}",
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        return $val;
    };
    (u16, $ptr: ident, $val: ident) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "xchg [{ptr}], {val:x}",
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        return $val;
    };
    (u32, $ptr: ident, $val: ident) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "xchg [{ptr}], {val:e}",
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        return $val;
    };
    (u64, $ptr: ident, $val: ident) => {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::asm!(
                "xchg [{ptr}], {val:r}",
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "lock; cmpxchg [{ptr}], {new_val}",
                // Load old_val into RAX as input/output register
                inout("al") $old_val,
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
                old_val = in(reg) $old_val,
                new_val = in(reg) $new_val,
                options(nostack)
            );
            $old_val = res;
        }

        return $old_val;
    };
    (u16, $ptr: ident, $old_val: ident, $new_val: ident) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "lock; cmpxchg [{ptr}], {new_val:x}",
                // Load old_val into RAX as input/output register
                inout("ax") $old_val,
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
                old_val = in(reg) $old_val,
                new_val = in(reg) $new_val,
                options(nostack)
            );
            $old_val = res;
        }

        return $old_val;
    };
    (u32, $ptr: ident, $old_val: ident, $new_val: ident) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!(
                "lock; cmpxchg [{ptr}], {new_val:e}",
                // Load old_val into RAX as input/output register
                inout("eax") $old_val,
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
                old_val = in(reg) $old_val,
                new_val = in(reg) $new_val,
                options(nostack)
            );
            $old_val = res;
        }

        return $old_val;
    };
    (u64, $ptr: ident, $old_val: ident, $new_val: ident) => {
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
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if $op == "add" {
                // The `add` operation can be optimized with XADD.
                core::arch::asm!(
                    "lock; xadd [{ptr}], {val}",
                    val = inout(reg_byte) $val,
                    ptr = in(reg) $ptr.as_ptr(),
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
                    ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        return $val;
    };
    (u16, $op: tt, $ptr: ident, $val: ident) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if $op == "add" {
                // The `add` operation can be optimized with XADD.
                core::arch::asm!(
                    "lock; xadd [{ptr}], {val:x}",
                    val = inout(reg) $val,
                    ptr = in(reg) $ptr.as_ptr(),
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
                    ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        return $val;
    };
    (u32, $op: tt, $ptr: ident, $val: ident) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if $op == "add" {
                // The `add` operation can be optimized with XADD.
                core::arch::asm!(
                    "lock; xadd [{ptr}], {val:e}",
                    val = inout(reg) $val,
                    ptr = in(reg) $ptr.as_ptr(),
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
                    ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
                val = in(reg) $val,
                options(nostack)
            );
            $val = res;
        }

        return $val;
    };
    (u64, $op: tt, $ptr: ident, $val: ident) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if $op == "add" {
                // The `add` operation can be optimized with XADD.
                core::arch::asm!(
                    "lock; xadd [{ptr}], {val:r}",
                    val = inout(reg) $val,
                    ptr = in(reg) $ptr.as_ptr(),
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
                    ptr = in(reg) $ptr.as_ptr(),
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
                ptr = in(reg) $ptr.as_ptr(),
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

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_load_8_seq_cst(ptr: NonNull<()>) -> u8 {
    gen_load!(u8, ptr, true);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_load_16_seq_cst(ptr: NonNull<()>) -> u16 {
    gen_load!(u16, ptr, true);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_load_32_seq_cst(ptr: NonNull<()>) -> u32 {
    gen_load!(u32, ptr, true);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
pub(crate) fn atomic_load_64_seq_cst(ptr: NonNull<()>) -> u64 {
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
pub(crate) fn atomic_load_8_unsynchronized(ptr: NonNull<()>) -> u8 {
    gen_load!(u8, ptr, false);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_load_16_unsynchronized(ptr: NonNull<()>) -> u16 {
    gen_load!(u16, ptr, false);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_load_32_unsynchronized(ptr: NonNull<()>) -> u32 {
    gen_load!(u32, ptr, false);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
pub(crate) fn atomic_load_64_unsynchronized(ptr: NonNull<()>) -> u64 {
    gen_load!(u64, ptr, false);
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_store_8_seq_cst(ptr: NonNull<()>, val: u8) {
    gen_store!(u8, ptr, val, true);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_store_16_seq_cst(ptr: NonNull<()>, val: u16) {
    gen_store!(u16, ptr, val, true);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_store_32_seq_cst(ptr: NonNull<()>, val: u32) {
    gen_store!(u32, ptr, val, true);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
pub(crate) fn atomic_store_64_seq_cst(ptr: NonNull<()>, val: u64) {
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
pub(crate) fn atomic_store_8_unsynchronized(ptr: NonNull<()>, val: u8) {
    gen_store!(u8, ptr, val, false);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_store_16_unsynchronized(ptr: NonNull<()>, val: u16) {
    gen_store!(u16, ptr, val, false);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_store_32_unsynchronized(ptr: NonNull<()>, val: u32) {
    gen_store!(u32, ptr, val, false);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
pub(crate) fn atomic_store_64_unsynchronized(ptr: NonNull<()>, val: u64) {
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
pub(crate) fn atomic_exchange_8_seq_cst(ptr: NonNull<()>, mut val: u8) -> u8 {
    gen_exchange!(u8, ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_exchange_16_seq_cst(ptr: NonNull<()>, mut val: u16) -> u16 {
    gen_exchange!(u16, ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_exchange_32_seq_cst(ptr: NonNull<()>, mut val: u32) -> u32 {
    gen_exchange!(u32, ptr, val);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
pub(crate) fn atomic_exchange_64_seq_cst(ptr: NonNull<()>, mut val: u64) -> u64 {
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
pub(crate) fn atomic_cmp_xchg_8_seq_cst(ptr: NonNull<()>, mut old_val: u8, new_val: u8) -> u8 {
    gen_cmpxchg!(u8, ptr, old_val, new_val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_cmp_xchg_16_seq_cst(ptr: NonNull<()>, mut old_val: u16, new_val: u16) -> u16 {
    gen_cmpxchg!(u16, ptr, old_val, new_val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_cmp_xchg_32_seq_cst(ptr: NonNull<()>, mut old_val: u32, new_val: u32) -> u32 {
    gen_cmpxchg!(u32, ptr, old_val, new_val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_cmp_xchg_64_seq_cst(ptr: NonNull<()>, mut old_val: u64, new_val: u64) -> u64 {
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
pub(crate) fn atomic_add_8_seq_cst(ptr: NonNull<()>, mut val: u8) -> u8 {
    gen_fetchop!(u8, "add", ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_add_16_seq_cst(ptr: NonNull<()>, mut val: u16) -> u16 {
    gen_fetchop!(u16, "add", ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_add_32_seq_cst(ptr: NonNull<()>, mut val: u32) -> u32 {
    gen_fetchop!(u32, "add", ptr, val);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
pub(crate) fn atomic_add_64_seq_cst(ptr: NonNull<()>, mut val: u64) -> u64 {
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
pub(crate) fn atomic_and_8_seq_cst(ptr: NonNull<()>, mut val: u8) -> u8 {
    gen_fetchop!(u8, "and", ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_and_16_seq_cst(ptr: NonNull<()>, mut val: u16) -> u16 {
    gen_fetchop!(u16, "and", ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_and_32_seq_cst(ptr: NonNull<()>, mut val: u32) -> u32 {
    gen_fetchop!(u32, "and", ptr, val);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
pub(crate) fn atomic_and_64_seq_cst(ptr: NonNull<()>, mut val: u64) -> u64 {
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
pub(crate) fn atomic_or_8_seq_cst(ptr: NonNull<()>, mut val: u8) -> u8 {
    gen_fetchop!(u8, "or", ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_or_16_seq_cst(ptr: NonNull<()>, mut val: u16) -> u16 {
    gen_fetchop!(u16, "or", ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_or_32_seq_cst(ptr: NonNull<()>, mut val: u32) -> u32 {
    gen_fetchop!(u32, "or", ptr, val);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
pub(crate) fn atomic_or_64_seq_cst(ptr: NonNull<()>, mut val: u64) -> u64 {
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
pub(crate) fn atomic_xor_8_seq_cst(ptr: NonNull<()>, mut val: u8) -> u8 {
    gen_fetchop!(u8, "xor", ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_xor_16_seq_cst(ptr: NonNull<()>, mut val: u16) -> u16 {
    gen_fetchop!(u16, "xor", ptr, val);
}
#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_xor_32_seq_cst(ptr: NonNull<()>, mut val: u32) -> u32 {
    gen_fetchop!(u32, "xor", ptr, val);
}

#[inline(always)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64",))]
pub(crate) fn atomic_xor_64_seq_cst(ptr: NonNull<()>, mut val: u64) -> u64 {
    gen_fetchop!(u64, "xor", ptr, val);
}

/// Size of a word (pointer) on this architecture.
pub(crate) const WORD_SIZE: usize = size_of::<usize>();
/// Number of words in a block (~cache line) on this architecture. Known to be
/// 8 on all supported architectures.
pub(crate) const WORDS_IN_BLOCK: usize = 8;
/// Size of a block (~cache line) on this architecture in bytes.
pub(crate) const BLOCK_SIZE: usize = WORD_SIZE * WORDS_IN_BLOCK;

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_copy_unaligned_block_down_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(u8, BLOCK_SIZE, "down");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_copy_unaligned_block_up_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(u8, BLOCK_SIZE, "up");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_copy_unaligned_word_down_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(u8, WORD_SIZE, "down");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_copy_unaligned_word_up_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(u8, WORD_SIZE, "up");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_copy_block_down_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(usize, WORDS_IN_BLOCK, "down");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_copy_block_up_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(usize, WORDS_IN_BLOCK, "up");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_copy_word_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(usize, 1, "down");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_copy32_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(u32, 1, "down");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_copy16_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(u16, 1, "down");
}

#[inline(always)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn atomic_copy8_unsynchronized(src: NonNull<()>, dst: NonNull<()>) {
    gen_copy!(u8, 1, "down");
}
