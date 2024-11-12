// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Artifact: mdbook-rendered HvLite Guide

/// Publish the artifact.
pub mod publish {
    use flowey::node::prelude::*;

    flowey_request! {
        pub struct Request {
            pub rendered_guide: ReadVar<PathBuf>,
            pub artifact_dir: ReadVar<PathBuf>,
            pub done: WriteVar<SideEffect>,
        }
    }

    new_simple_flow_node!(struct Node);

    impl SimpleFlowNode for Node {
        type Request = Request;

        fn imports(ctx: &mut ImportCtx<'_>) {
            ctx.import::<flowey_lib_common::copy_to_artifact_dir::Node>();
        }

        fn process_request(request: Self::Request, ctx: &mut NodeCtx<'_>) -> anyhow::Result<()> {
            let Request {
                rendered_guide,
                artifact_dir,
                done,
            } = request;

            let files = rendered_guide.map(ctx, |p| vec![("Guide".into(), p)]);
            ctx.req(flowey_lib_common::copy_to_artifact_dir::Request {
                debug_label: "guide".into(),
                files,
                artifact_dir,
                done,
            });

            Ok(())
        }
    }
}

/// Resolve the contents of an existing artifact.
pub mod resolve {
    use flowey::node::prelude::*;

    flowey_request! {
        pub struct Request {
            pub artifact_dir: ReadVar<PathBuf>,
            pub rendered_guide: WriteVar<PathBuf>,
        }
    }

    new_simple_flow_node!(struct Node);

    impl SimpleFlowNode for Node {
        type Request = Request;

        fn imports(ctx: &mut ImportCtx<'_>) {
            ctx.import::<flowey_lib_common::copy_to_artifact_dir::Node>();
        }

        fn process_request(request: Self::Request, ctx: &mut NodeCtx<'_>) -> anyhow::Result<()> {
            let Request {
                artifact_dir,
                rendered_guide,
            } = request;

            ctx.emit_rust_step("resolve guide artifact", |ctx| {
                let artifact_dir = artifact_dir.claim(ctx);
                let rendered_guide = rendered_guide.claim(ctx);
                move |rt| {
                    let artifact_dir = rt.read(artifact_dir);
                    let guide_dir = artifact_dir.join("Guide");
                    if !guide_dir.exists() {
                        anyhow::bail!("malformed artifact! did not find Guide/ folder");
                    }
                    rt.write(rendered_guide, &guide_dir);
                    Ok(())
                }
            });

            Ok(())
        }
    }
}
