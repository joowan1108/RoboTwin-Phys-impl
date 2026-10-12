"""Infrastructure smoke test only. Not a learned policy or benchmark baseline."""
def get_model(args):
    return None


def reset_model(model):
    pass


def eval(env, model, observation):
    env.take_action(observation["joint_action"]["vector"])
