// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Download various pre-built `openvmm-deps` dependencies.

use flowey::node::prelude::*;
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OpenvmmDepsArch {
    X86_64,
    Aarch64,
}

flowey_request! {
    pub enum Request {
        /// Specify version of the github release to pull from
        Version(String),
        GetLinuxTestKernel(OpenvmmDepsArch, WriteVar<PathBuf>),
        GetLinuxTestInitrd(OpenvmmDepsArch, WriteVar<PathBuf>),
        GetOpenhclCpioDbgrd(OpenvmmDepsArch, WriteVar<PathBuf>),
        GetOpenhclCpioShell(OpenvmmDepsArch, WriteVar<PathBuf>),
        GetOpenhclSysroot(OpenvmmDepsArch, WriteVar<PathBuf>),
    }
}

new_flow_node!(struct Node);

impl FlowNode for Node {
    type Request = Request;

    fn imports(ctx: &mut ImportCtx<'_>) {
        ctx.import::<flowey_lib_common::install_dist_pkg::Node>();
        ctx.import::<flowey_lib_common::download_gh_release::Node>();
    }

    fn emit(requests: Vec<Self::Request>, ctx: &mut NodeCtx<'_>) -> anyhow::Result<()> {
        let mut version = None;
        let mut linux_test_kernel: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut linux_test_initrd: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut openhcl_cpio_dbgrd: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut openhcl_cpio_shell: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut openhcl_sysroot: BTreeMap<_, Vec<_>> = BTreeMap::new();

        for req in requests {
            match req {
                Request::Version(v) => same_across_all_reqs("Version", &mut version, v)?,

                Request::GetLinuxTestKernel(arch, var) => {
                    linux_test_kernel.entry(arch).or_default().push(var)
                }
                Request::GetLinuxTestInitrd(arch, var) => {
                    linux_test_initrd.entry(arch).or_default().push(var)
                }
                Request::GetOpenhclCpioDbgrd(arch, var) => {
                    openhcl_cpio_dbgrd.entry(arch).or_default().push(var)
                }
                Request::GetOpenhclCpioShell(arch, var) => {
                    openhcl_cpio_shell.entry(arch).or_default().push(var)
                }
                Request::GetOpenhclSysroot(arch, var) => {
                    openhcl_sysroot.entry(arch).or_default().push(var)
                }
            }
        }

        let version = version.ok_or(anyhow::anyhow!("Missing essential request: Version"))?;

        // -- end of req processing -- //

        if linux_test_kernel.is_empty()
            && linux_test_initrd.is_empty()
            && openhcl_cpio_dbgrd.is_empty()
            && openhcl_cpio_shell.is_empty()
            && openhcl_sysroot.is_empty()
        {
            return Ok(());
        }

        let extract_tar_bz2_deps =
            flowey_lib_common::_util::extract::extract_tar_bz2_if_new_deps(ctx);

        let openvmm_deps_tar_bz2_x64 = if linux_test_initrd.contains_key(&OpenvmmDepsArch::X86_64)
            || linux_test_kernel.contains_key(&OpenvmmDepsArch::X86_64)
            || openhcl_cpio_dbgrd.contains_key(&OpenvmmDepsArch::X86_64)
            || openhcl_cpio_shell.contains_key(&OpenvmmDepsArch::X86_64)
            || openhcl_sysroot.contains_key(&OpenvmmDepsArch::X86_64)
        {
            Some(
                ctx.reqv(|v| flowey_lib_common::download_gh_release::Request {
                    repo_owner: "microsoft".into(),
                    repo_name: "openvmm-deps".into(),
                    needs_auth: false,
                    tag: version.clone(),
                    file_name: format!("openvmm-deps.x86_64.{version}.tar.bz2"),
                    path: v,
                }),
            )
        } else {
            None
        };

        let openvmm_deps_tar_bz2_aarch64 = if linux_test_initrd
            .contains_key(&OpenvmmDepsArch::Aarch64)
            || linux_test_kernel.contains_key(&OpenvmmDepsArch::Aarch64)
            || openhcl_cpio_dbgrd.contains_key(&OpenvmmDepsArch::Aarch64)
            || openhcl_cpio_shell.contains_key(&OpenvmmDepsArch::Aarch64)
            || openhcl_sysroot.contains_key(&OpenvmmDepsArch::Aarch64)
        {
            Some(
                ctx.reqv(|v| flowey_lib_common::download_gh_release::Request {
                    repo_owner: "microsoft".into(),
                    repo_name: "openvmm-deps".into(),
                    needs_auth: false,
                    tag: version.clone(),
                    file_name: format!("openvmm-deps.aarch64.{version}.tar.bz2"),
                    path: v,
                }),
            )
        } else {
            None
        };

        ctx.emit_rust_step("unpack openvmm-deps archive", |ctx| {
            let extract_tar_bz2_deps = extract_tar_bz2_deps.claim(ctx);
            let openvmm_deps_tar_bz2_x64 = openvmm_deps_tar_bz2_x64.claim(ctx);
            let openvmm_deps_tar_bz2_aarch64 = openvmm_deps_tar_bz2_aarch64.claim(ctx);

            let linux_test_kernel = linux_test_kernel.claim(ctx);
            let linux_test_initrd = linux_test_initrd.claim(ctx);
            let openhcl_cpio_dbgrd = openhcl_cpio_dbgrd.claim(ctx);
            let openhcl_cpio_shell = openhcl_cpio_shell.claim(ctx);
            let openhcl_sysroot = openhcl_sysroot.claim(ctx);
            move |rt| {
                let extract_dir_x64 = openvmm_deps_tar_bz2_x64
                    .map(|file| {
                        let file = rt.read(file);
                        flowey_lib_common::_util::extract::extract_tar_bz2_if_new(
                            rt,
                            extract_tar_bz2_deps.clone(),
                            &file,
                            &version,
                        )
                    })
                    .transpose()?;
                let extract_dir_aarch64 = openvmm_deps_tar_bz2_aarch64
                    .map(|file| {
                        let file = rt.read(file);
                        flowey_lib_common::_util::extract::extract_tar_bz2_if_new(
                            rt,
                            extract_tar_bz2_deps.clone(),
                            &file,
                            &version,
                        )
                    })
                    .transpose()?;

                let base_dir = move |arch| match arch {
                    OpenvmmDepsArch::X86_64 => extract_dir_x64.clone().unwrap(),
                    OpenvmmDepsArch::Aarch64 => extract_dir_aarch64.clone().unwrap(),
                };

                let kernel_file_name = |arch| match arch {
                    OpenvmmDepsArch::X86_64 => "vmlinux",
                    OpenvmmDepsArch::Aarch64 => "Image",
                };

                for (arch, vars) in linux_test_kernel {
                    let path = base_dir(arch).join(kernel_file_name(arch));
                    rt.write_all(vars, &path)
                }

                for (arch, vars) in linux_test_initrd {
                    let path = base_dir(arch).join("initrd");
                    rt.write_all(vars, &path)
                }

                for (arch, vars) in openhcl_cpio_dbgrd {
                    let path = base_dir(arch).join("dbgrd.cpio.gz");
                    rt.write_all(vars, &path)
                }

                for (arch, vars) in openhcl_cpio_shell {
                    let path = base_dir(arch).join("shell.cpio.gz");
                    rt.write_all(vars, &path)
                }

                for (arch, vars) in openhcl_sysroot {
                    let path = base_dir(arch).join("sysroot.tar.gz");
                    rt.write_all(vars, &path)
                }

                Ok(())
            }
        });

        Ok(())
    }
}
