from ..model import RobotModel


class RobotModelFromURDF(RobotModel):

    def __init__(self, urdf=None, urdf_file=None, global_scale=1.0):
        super(RobotModelFromURDF, self).__init__()

        if urdf and urdf_file:
            raise ValueError(
                "'urdf' and 'urdf_file' cannot be given at the same time"
            )
        if urdf:
            self.load_urdf(urdf=urdf, global_scale=global_scale)
        elif urdf_file:
            self.load_urdf_file(file_obj=urdf_file, global_scale=global_scale)
        else:
            self.load_urdf_file(file_obj=self.default_urdf_path)

    @property
    def default_urdf_path(self):
        raise NotImplementedError
