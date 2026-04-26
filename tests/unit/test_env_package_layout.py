from env.observations import ObservationBuilder, SwarmObserver
from env.oracles import OracleManager, build_distance_field
from env.physics import GlobalOUWind, PhysicsCore, PhysicsLoop
from env.rewards import COST_KEYS, RewardConfig, SwarmRewarder
from env.scenes import ForestConfig, SceneManager, SpawnController, threat_from_config


def test_env_subpackages_export_expected_symbols():
    assert ObservationBuilder is not None
    assert SwarmObserver is not None
    assert OracleManager is not None
    assert build_distance_field is not None
    assert PhysicsCore is not None
    assert PhysicsLoop is not None
    assert GlobalOUWind is not None
    assert RewardConfig is not None
    assert SwarmRewarder is not None
    assert COST_KEYS
    assert ForestConfig is not None
    assert SceneManager is not None
    assert SpawnController is not None
    assert threat_from_config is not None
