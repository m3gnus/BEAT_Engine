@testset "condensed backend request and cache agreement" begin
    fem = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
    bem = load_gmsh22_with_tags(joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"), 0.001)
    map = build_conforming_interface_map(fem, bem, physical_tag(fem, 2, "Interface"), 2)
    cache = prepare_condensed_coupled_cache(fem, bem, map;
        quadrature_order=CONDENSED_QUADRATURE_ORDER, singular_order=CONDENSED_SINGULAR_ORDER,
        bem_backend=:cpu)
    @test_throws "cache backend does not match" build_condensed_coupled_system(fem, bem, map, 500.0, 343.0, 1.21;
        cache=cache, bem_backend=:metal)
    @test_throws "backend must be :cpu or :metal" build_condensed_coupled_system(fem, bem, map, 500.0, 343.0, 1.21;
        bem_backend=:unsupported)
end
