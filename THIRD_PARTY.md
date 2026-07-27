# Third-party foundations

This reproduction follows the public [TopoLogic](https://github.com/Franpin/TopoLogic)
architecture and OpenLane-V2 evaluation protocol. TopoLogic and the OpenLane-V2
devkit are Apache-2.0 licensed. The experiment downloads, but does not redistribute,
the CC BY-NC-SA 4.0 OpenLane-V2 public sample and the public Cityscapes-trained
SegFormer checkpoint.

The compact decoder is a clean-room, bounded implementation for current Blackwell
GPUs. It preserves the matched DETR-style centerline decoder and tests only the two
mechanism changes disclosed in HGeo-TopoMap: distance-masked road-prior attention
(GAL) and orientation-group contrastive consistency (GCL).

