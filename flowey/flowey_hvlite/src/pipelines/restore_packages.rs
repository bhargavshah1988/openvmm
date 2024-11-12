// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

use flowey::node::prelude::ReadVar;
use flowey::pipeline::prelude::*;
use flowey_lib_hvlite::run_cargo_build::common::CommonArch;

#[derive(clap::ValueEnum, Clone, Copy)]
pub enum CommonArchCli {
    X86_64,
    Aarch64,
}

impl From<CommonArchCli> for CommonArch {
    fn from(value: CommonArchCli) -> Self {
        match value {
            CommonArchCli::X86_64 => CommonArch::X86_64,
            CommonArchCli::Aarch64 => CommonArch::Aarch64,
        }
    }
}

#[derive(clap::Args)]
/// Download and restore packages needed for building the specified architectures.
pub struct RestorePackagesCli {
    arch: Vec<CommonArchCli>,
}

impl IntoPipeline for RestorePackagesCli {
    fn into_pipeline(self, backend_hint: PipelineBackendHint) -> anyhow::Result<Pipeline> {
        let openvmm_repo = flowey_lib_common::git_checkout::RepoSource::ExistingClone(
            ReadVar::from_static(crate::repo_root()),
        );

        let mut pipeline = Pipeline::new();
        let mut job = pipeline
            .new_job(
                FlowPlatform::host(backend_hint),
                FlowArch::host(backend_hint),
                "restore packages",
            )
            .dep_on(|_| flowey_lib_hvlite::_jobs::cfg_versions::Request {})
            .dep_on(
                |_| flowey_lib_hvlite::_jobs::cfg_hvlite_reposource::Params {
                    hvlite_repo_source: openvmm_repo,
                },
            )
            .dep_on(|_| flowey_lib_hvlite::_jobs::cfg_common::Params {
                local_only: Some(flowey_lib_hvlite::_jobs::cfg_common::LocalOnlyParams {
                    interactive: true,
                    auto_install: true,
                    force_nuget_mono: false,
                    external_nuget_auth: false,
                    ignore_rust_version: true,
                }),
                verbose: ReadVar::from_static(true),
                locked: false,
                deny_warnings: false,
            });

        let arches = {
            if self.arch.is_empty() {
                vec![match FlowArch::host(backend_hint) {
                    FlowArch::X86_64 => CommonArchCli::X86_64,
                    FlowArch::Aarch64 => CommonArchCli::Aarch64,
                    arch => anyhow::bail!("unsupported arch {arch}"),
                }]
            } else {
                self.arch
            }
        };

        for arch in arches {
            job = job.dep_on(
                |ctx| flowey_lib_hvlite::_jobs::local_restore_packages::Request {
                    arch: arch.into(),
                    done: ctx.new_done_handle(),
                },
            );
        }
        job.finish();
        Ok(pipeline)
    }
}
