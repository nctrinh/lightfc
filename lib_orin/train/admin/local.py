class EnvironmentSettings:
    def __init__(self, env_num=0):
        self.workspace_dir = r'/mnt/tupm/LightFC'  # Base directory for saving network checkpoints.
        self.tensorboard_dir = r'/mnt/tupm/LightFC/tensorboard'  # Directory for tensorboard files.
        self.pretrained_networks = r'/mnt/tupm/LightFC/pretrained_models'

        self.lasot_dir = '/mnt/tupm/data/lasot'
        self.got10k_dir = '/mnt/tupm/data/got10k/train'
        self.got10k_val_dir = '/mnt/tupm/data/got10k/val'
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_dir = ''
        self.trackingnet_lmdb_dir = ''
        self.coco_dir = '/mnt/tupm/data/coco'
        self.coco_lmdb_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''

        self.imagenet_dir = ''
        self.imagenet_lmdb_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''

        self.uav123_dir = '/mnt/tupm/data/uav123'
        self.dronestagram_dir = '/mnt/tupm/data/drone-bd/Dronestagram'
        self.geotext_dir = '/mnt/tupm/data/GeoText1652'
        self.openlogo_dir = '/mnt/huydd/code/Cross-view_Image_Retrieval/data/openlogo'