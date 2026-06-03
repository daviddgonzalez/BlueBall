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


def test_camera_scale_defaults_to_one_and_affects_world_to_screen():
    cam = Camera(viewport_w=200, viewport_h=100)
    assert cam.scale == 1.0
    # At scale 1, world (0,0) with camera position (0,0) → viewport center.
    assert cam.world_to_screen((0.0, 0.0)) == (100.0, 50.0)
    cam.scale = 2.0
    # Doubling scale doubles the per-unit offset.
    assert cam.world_to_screen((10.0, 0.0)) == (120.0, 50.0)


def test_free_camera_zoom_keys_change_scale_within_bounds():
    import pygame
    import pytest
    from blueball.camera import FreeCamera
    cam = FreeCamera(viewport_w=200, viewport_h=100)
    initial = cam.scale
    events = [pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_EQUALS})]
    cam.handle_events(events)
    assert cam.scale == pytest.approx(initial * cam.ZOOM_STEP)
    # Repeatedly zoom out below the floor: scale must clamp at ZOOM_MIN.
    for _ in range(50):
        cam.handle_events([pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_MINUS})])
    assert cam.scale == pytest.approx(cam.ZOOM_MIN)


def test_free_camera_arrow_keys_pan_position():
    import pygame
    from blueball.camera import FreeCamera
    cam = FreeCamera(viewport_w=200, viewport_h=100)
    # Mock keys_pressed: right + down held.
    class _Keys:
        def __init__(self, held):
            self._held = set(held)
        def __getitem__(self, key):
            return key in self._held
    keys = _Keys({pygame.K_RIGHT, pygame.K_DOWN})
    cam.update(keys_pressed=keys, dt=1.0)
    px, py = cam.position
    assert px > 0       # camera moved right (positive world-x)
    assert py > 0       # camera moved down (positive world-y in our y-down coords)
