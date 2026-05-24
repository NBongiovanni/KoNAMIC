import cv2
import numpy as np
import torch


class VisionObserver:
    def __init__(self, state_renderer, drone_dim: int):
        self.state_renderer = state_renderer
        self.drone_dim = drone_dim

    def observe(self, x_k: np.ndarray, debug: bool = False) -> torch.Tensor:
        rendered = self.state_renderer.pipeline(np.asarray(x_k))

        if self.drone_dim == 2:
            im = rendered.astype(np.float32) / 255.0
            if im.ndim == 2:
                im = np.expand_dims(im, axis=0)  # [1, H, W]

            if debug:
                cv2.imshow("Rendered state", im[0])
                key = cv2.waitKey(0) & 0xFF
                if key in (27, ord('q')):
                    cv2.destroyAllWindows()
                    raise KeyboardInterrupt

            return torch.from_numpy(im)

        elif self.drone_dim == 3:
            img_left, img_right = rendered
            img_left = img_left.astype(np.float32) / 255.0
            img_right = img_right.astype(np.float32) / 255.0

            if img_left.ndim == 2:
                img_left = np.expand_dims(img_left, axis=0)
            if img_right.ndim == 2:
                img_right = np.expand_dims(img_right, axis=0)

            if debug:
                cv2.imshow("Rendered left", img_left[0])
                cv2.imshow("Rendered right", img_right[0])
                key = cv2.waitKey(0) & 0xFF
                if key in (27, ord('q')):
                    cv2.destroyAllWindows()
                    raise KeyboardInterrupt

            im = np.concatenate([img_left, img_right], axis=0)  # [2, H, W]
            return torch.from_numpy(im)

        else:
            raise ValueError(f"Drone dimension inconnue: {self.drone_dim}")