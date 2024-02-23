using ClosedLoopReachability, Test
import Aqua

@testset "Aqua tests" begin
    Aqua.test_all(ClosedLoopReachability; ambiguities=false)

    # do not warn about ambiguities in dependencies
    Aqua.test_ambiguities(ClosedLoopReachability)
end
