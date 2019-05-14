from dm_control import suite
from jaco import jaco

import inspect
import dm_control2gym

LOCAL_DOMAINS = {name: module for name, module in locals().items()
                 if inspect.ismodule(module) and hasattr(module, 'SUITE')}

suite._DOMAINS = {**suite._DOMAINS, **LOCAL_DOMAINS}

env = dm_control2gym.make(domain_name="jaco", task_name="basic")
