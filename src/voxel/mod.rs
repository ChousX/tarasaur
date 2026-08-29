pub mod buffers;
pub mod pipeline;
pub mod systems;
pub mod types;

use bevy::{
    asset::{load_internal_asset, uuid_handle},
    core_pipeline::{Core3d, Core3dSystems},
    prelude::*,
    render::{Render, RenderApp, RenderSystems},
};

use pipeline::{VoxelComputePipeline, VoxelPipelineLayouts};
use systems::{dispatch_voxel_compute_passes, extract_voxel_chunks, prepare_voxel_chunk_buffers};

use crate::voxel::{
    pipeline::{VoxelDummyMaterial, VoxelRasterPipeline},
    systems::voxel_raster_pass,
};

pub const SURFACE_NETS_PASS1_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("9f3a1b2c-4d5e-6f70-8192-a3b4c5d6e7f8"); // any valid UUIDv4
pub const STREAM_COMPACTION_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("1a2b3c4d-5e6f-7081-92a3-b4c5d6e7f809");
pub const SURFACE_NETS_PASS3_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("2b3c4d5e-6f70-8192-a3b4-c5d6e7f8091a");
pub const VOXEL_RASTER_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("3c4d5e6f-7081-92a3-b4c5-d6e7f8091a2b");

pub struct VoxelRenderPlugin;

impl Plugin for VoxelRenderPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            SURFACE_NETS_PASS1_SHADER_HANDLE,
            "shaders/surface_nets_pass1.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            STREAM_COMPACTION_SHADER_HANDLE,
            "shaders/stream_compaction.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            SURFACE_NETS_PASS3_SHADER_HANDLE,
            "shaders/surface_nets_pass3.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            VOXEL_RASTER_SHADER_HANDLE,
            "shaders/voxel_raster.wgsl",
            Shader::from_wgsl
        );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(ExtractSchedule, extract_voxel_chunks)
            .add_systems(
                Render,
                (
                    prepare_voxel_chunk_buffers.in_set(RenderSystems::Prepare),
                    dispatch_voxel_compute_passes.in_set(RenderSystems::Queue),
                ),
            )
            .add_systems(Core3d, voxel_raster_pass.in_set(Core3dSystems::MainPass));
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<VoxelRasterPipeline>()
            .init_resource::<VoxelDummyMaterial>()
            .init_resource::<VoxelPipelineLayouts>()
            .init_resource::<VoxelComputePipeline>();
    }
}
