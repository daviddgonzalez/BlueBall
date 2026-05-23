from blueball import config
from blueball.camera import Camera, FollowCamera


def test_world_to_screen_centers_on_camera_position():
    cam = Camera(viewport_w=800, viewport_h=600)
    cam.position = (1000.0, 500.0)
    # World point exactly at camera position should be screen center
    sx, sy = cam.world_to_screen((1000.0, 500.0))
    assert sx == 400.0
    assert sy == 300.0


def test_world_to_screen_offsets_correctly():
    cam = Camera(viewport_w=800, viewport_h=600)
    cam.position = (1000.0, 500.0)
    # World point 100 right of camera -> screen point 100 right of center
    sx, sy = cam.world_to_screen((1100.0, 500.0))
    assert sx == 500.0
    assert sy == 300.0


def test_follow_camera_stays_still_inside_deadzone():
    cam = FollowCamera(viewport_w=800, viewport_h=600)
    cam.position = (1000.0, 500.0)
    half_w = config.CAMERA_DEAD_ZONE_W / 2
    # Move target by less than the dead-zone half-width
    cam.update(target=(1000.0 + half_w - 1, 500.0), dt=1 / 60)
    assert cam.position == (1000.0, 500.0)


def test_follow_camera_lerps_when_target_leaves_deadzone():
    cam = FollowCamera(viewport_w=800, viewport_h=600)
    cam.position = (1000.0, 500.0)
    half_w = config.CAMERA_DEAD_ZONE_W / 2
    # Move target outside the dead-zone to the right
    target = (1000.0 + half_w + 50, 500.0)
    initial_x = cam.position[0]
    cam.update(target=target, dt=1 / 60)
    # Camera should have moved right, but not all the way to the target
    assert cam.position[0] > initial_x
    assert cam.position[0] < target[0]
