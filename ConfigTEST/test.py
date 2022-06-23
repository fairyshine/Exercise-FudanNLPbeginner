from omegaconf import DictConfig, OmegaConf
import hydra


#参数管理的3种方法

# 1. Hydra + omegaconf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.basic.another.value4)


# 2. omegaconf
basic = OmegaConf.load('conf/basic/basic.yaml')

# 3. 参数类
class localConfig(object):
    """配置参数"""
    def __init__(self):
        self.value1=2
        self.value2='c'


if __name__ == "__main__":
    localConfig=localConfig()
    print("法3：")
    print('localconfig.value1: ',localConfig.value1)
    print('localconfig.value2: ',localConfig.value2)
    print("法2：")
    print('omegaconf————basic.another.value3: ',basic.another.value3)
    print("法1：")
    main()