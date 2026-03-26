using ERGMMulti
using Graphs
using Test

@testset "ERGMMulti.jl" begin
    @testset "Module loading" begin
        @test @isdefined(ERGMMulti)
    end

    @testset "MultilayerNetwork construction" begin
        mln = MultilayerNetwork(10)
        @test mln isa MultilayerNetwork{Int}
        @test mln.n_vertices == 10

        mln2 = MultilayerNetwork(5; layer_names=[:friendship, :advice])
        @test length(mln2.layers) == 2
        @test haskey(mln2.layers, :friendship)
        @test haskey(mln2.layers, :advice)
    end

    @testset "MultilayerNetwork operations" begin
        mln = MultilayerNetwork(5; layer_names=[:a, :b])
        @test Graphs.nv(mln) == 5
    end

    @testset "Multilayer ERGM terms" begin
        @test LayerEdges(:friendship) isa LayerEdges
        @test LayerMutual(:advice) isa LayerMutual
        @test LayerTriangle(:trust) isa LayerTriangle
        @test MultiplexMutual(:a, :b) isa MultiplexMutual
        @test InterlayerDependence(:a, :b) isa InterlayerDependence
        @test CrossNetEdges() isa CrossNetEdges
        @test BetweenLayers(:a, :b) isa BetweenLayers
    end

    @testset "Multilevel terms" begin
        @test Nestedness(1) isa Nestedness
        @test CrossLevelEdge(1, 2) isa CrossLevelEdge
        @test LevelHomophily(1) isa LevelHomophily
    end

    @testset "Estimation API" begin
        @test ergm_multi === fit_multi_ergm
    end

    @testset "Utilities" begin
        @test isdefined(ERGMMulti, :as_multilayer)
        @test isdefined(ERGMMulti, :combine_networks)
        @test isdefined(ERGMMulti, :split_by_layer)
    end

    @testset "Simulation" begin
        @test isdefined(ERGMMulti, :simulate_multi_ergm)
    end
end
