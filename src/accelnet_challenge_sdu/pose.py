import numpy as np
import quaternion  # https://github.com/moble/quaternion
import geometry_msgs.msg
from .rotations import Rx, Ry, Rz


class Pose:
    """A pose represented by a translation vector and a quaternion.
    """
    def __init__(self, translation=None, rotation=None):
        self.p = np.zeros(3) if translation is None else np.array(translation)

        if isinstance(rotation, np.quaternion):
            self.q = rotation.copy()
        elif rotation is None:
            self.q = np.quaternion(1, 0, 0, 0)
        else:
            self.q = np.quaternion(*rotation)

    def __repr__(self):
        return f'Pose(translation={self.p}, rotation={self.q}'

    def __matmul__(self, rhs):
        # Return a new Pose with translation
        #   self.p + rhs.p'
        # where rhs.p' is the vector rhs.p rotated by self.q (by conjugation),
        # and rotation composed by
        #   self.q * rhs.q
        if not isinstance(rhs, Pose):
            raise TypeError(f"TypeError: unsupported operand type for @: '{type(rhs).__name__}'")

        return Pose(self.p + (self.q * np.quaternion(0, *rhs.p) * self.q.conjugate()).vec, self.q * rhs.q)

    def inverse(self):
        return Pose(-(self.q.conjugate() * np.quaternion(0, *self.p) * self.q).vec, self.q.conjugate())

    def to_matrix(self):
        T = np.zeros((4, 4))
        T[:3,:3] = quaternion.as_rotation_matrix(self.q)
        T[:3,3] = self.p
        T[3,3] = 1
        return T

    @staticmethod
    def from_matrix(T):
        return Pose(T[:3,3], quaternion.from_rotation_matrix(T[:3,:3]))

    @staticmethod
    def from_msg(m):
        if m._type == 'geometry_msgs/Transform':
            return Pose(np.array([m.translation.x, m.translation.y, m.translation.z]),
                        np.quaternion(m.rotation.w, m.rotation.x, m.rotation.y, m.rotation.z))
        elif m._type == 'geometry_msgs/Pose':
            return Pose(np.array([m.position.x, m.position.y, m.position.z]),
                        np.quaternion(m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z))
        else:
            raise TypeError(f"Bad message type: '{type(m)}'")

    def to_msg(self, msg_type):
        if msg_type == 'geometry_msgs/Transform':
            return geometry_msgs.msg.Transform(
                geometry_msgs.msg.Vector3(*self.p),
                geometry_msgs.msg.Quaternion(self.q.x, self.q.y, self.q.z, self.q.w))
        elif msg_type == 'geometry_msgs/Pose':
            return geometry_msgs.msg.Pose(
                geometry_msgs.msg.Point(*self.p),
                geometry_msgs.msg.Quaternion(self.q.x, self.q.y, self.q.z, self.q.w))
        else:
            raise TypeError(f"Bad message type: '{msg_type}'")

    def astuple(self):
        return (self.p, self.q)

    @staticmethod
    def Rx(angle):
        return Pose.from_matrix(Rx(angle))

    @staticmethod
    def Ry(angle):
        return Pose.from_matrix(Ry(angle))

    @staticmethod
    def Rz(angle):
        return Pose.from_matrix(Rz(angle))


if __name__ == '__main__':
    pose1 = Pose(
        np.array([1, 3, 2]),
        quaternion.from_euler_angles([np.pi/7, np.pi/3, np.pi/3]))

    pose2 = Pose(
        np.array([5, 0, 3]),
        quaternion.from_euler_angles([np.pi/3, np.pi, 0]))

    pose3 = Pose(np.array([4.2, 6.6, 1.3]),
            quaternion.from_euler_angles([np.pi, 0, np.pi/5]))

    T1 = pose1.to_matrix()
    T2 = pose2.to_matrix()
    T3 = pose3.to_matrix()
    T4 = np.eye(4)
    T4[0,3] = -0.1

    assert np.allclose(T2 @ T3 @ T2, (pose2 @ pose3 @ pose2).to_matrix())
    assert np.allclose(T3 @ T3 @ T1, (pose3 @ pose3 @ pose1).to_matrix())
    assert np.allclose(T3 @ T1 @ np.linalg.inv(T3), (pose3 @ pose1 @ pose3.inverse()).to_matrix())
    assert np.allclose(np.linalg.inv(T3) @ T1 @ T2, (pose3.inverse() @ pose1 @ pose2).to_matrix())
    assert np.allclose(np.eye(4), Pose().to_matrix())
    assert np.allclose(Pose().p, Pose.from_matrix(np.eye(4)).p)
    assert quaternion.isclose(Pose().q, Pose.from_matrix(np.eye(4)).q)
    assert np.allclose(pose3.inverse().p, Pose.from_matrix(np.linalg.inv(T3)).p)
    assert quaternion.isclose(pose3.inverse().q, Pose.from_matrix(np.linalg.inv(T3)).q)
    assert np.allclose(T1 @ T4, (pose1 @ Pose([-0.1, 0, 0])).to_matrix())

    import timeit
    print(timeit.timeit('T1 @ T2 @ T3', setup='from __main__ import T1, T2, T3', number=10000))
    print(timeit.timeit('pose1 @ pose2 @ pose3', setup='from __main__ import pose1, pose2, pose3', number=10000))
    print(timeit.timeit('Pose.from_matrix(T3)', setup='from __main__ import Pose, T3', number=10000))
    print(timeit.timeit('pose3.to_matrix()', setup='from __main__ import pose3', number=100000))
