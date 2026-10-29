"""
utilities for benchmarking the pipelines
"""
import json
import os
import time

def save_benchmark(pipeline_type, data_type, benchmarks):
    """
    benchmarks should follow the structure found in the bottom of this script
    """

    local_time = time.localtime() #to get day-hour-min

    if not os.path.isdir("benchmarks"):
        os.mkdir("benchmarks")

    with open(f"benchmarks/{pipeline_type}-{data_type}-benchmark-{local_time.tm_mday}-{local_time.tm_hour}-{local_time.tm_min}.json", "w") as f: #timestamp at the end in case we want multiple benchmarks of the same pipeline/data
        json.dump(
            {
                benchmarks
            },
            f,
            indent=4,
        )

"""
TODO WIP, still need to discuss what is valuable to benchmark

benchmarks = {
    "data":{ #this block is to identify the data
        "data type": data_type, #ps2d, betti or ps2d+betti
        "load type", #full realization or random realization
        "shape of obs", #shape of given data type's observation
        "number of obs", #number of observations
        "shape of theta", #shape of the theta parameter
        "number of theta",
        "pipeline", #either singular or ensemble
    },
    "benchmarks":{
        "gridsearching":{
            "average time per parameter",
            "max time for a parameter",
            "min time for a parameter",
            "average cv time",
            "max cv time",
            "min cv time",
            "total gridsearching time",
        },
        "model training":{
            "training time":{
                "average time",
                "max training time",
                "min training time",
                "total time", #to train all 30 models
            },
            "data preparation":{
                "average load time",
                "max load time",
                "min load time",
                "total time",
            },
            "embedding":{
                "betti":{ #only for betti
                    "forward pass time"
                },
                "resnet":{ #only for ps2d and ps2d+betti
                    "average epoch time",
                    "total time",
                },
            }
            "density estimator training":{
                "average one model",
                "max ",
            },
            "TARP":{
                "20 TARP min time",
                "20 TARP max time",
                "20 TARP average time for one realization",
                "20 TARP total time",
                "average time for the 20 realizations (for the 30 models)",
                "TARP draw figure time",
            },
        "model testing": {
            "data preparation":{
                "average load time",
                "max load time",
                "min load time",
                "total time",
            },
            "sampling theta_pred":{
                "average sample time",
                "max sample time",
                "min sample time",
                "total time",
            },
            "model metrics":{
                "per bin":{
                    "average load time",
                    "max load time",
                    "min load time",
                    "total time",
                },
                "overall metrics time",
            },
            "reionization history and pairplots":{
                "low time",
                "mid time",
                "high time",
                "average time",
                "min time",
                "max time",
                "total time",
            },
            "uncertainty":{
                "low time",
                "mid time",
                "high time",
                "average time",
                "min time",
                "max time",
                "total time",
            }
            "global TARP":{
                "20 TARP min time",
                "20 TARP max time",
                "20 TARP average time for one realization",
                "20 TARP total time",
                "average time for the 20 realizations (for the 30 models)",
                "TARP draw figure time",
            },
            "PPC":{
                "low PPC time",
                "mid PPC time",
                "high PPC time",
                "average PPC time",
                "min PPC time",
                "max PPC time",
                "total PPC time",
            },
            "total testing time",
        },
        "total pipeline time",
    }
}
"""
