import imageio
import os


class VideoRecorder(object):
    def __init__(self,
                 env,
                 enabled=True,
                 height=256,
                 width=256,
                 camera_id=0,
                 fps=30):
        self._env = env
        self._enabled = enabled
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._fps = fps
        self._frames = []

    def record(self):
        if self._enabled:
            frame = self._env.render(
                mode='rgb_array',
                # height=self._height,
                # width=self._width,
                # camera_id=self._camera_id
            )
            self._frames.append(frame.copy())

    def save(self, dir_name, file_name):
        if self._enabled:
            path = os.path.join(dir_name, file_name)
            imageio.mimsave(path, self._frames, fps=self._fps)
