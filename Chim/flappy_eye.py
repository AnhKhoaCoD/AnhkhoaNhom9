import pygame, random, time, threading, cv2, os, sys
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pygame.locals import *

# VARIABLES
SCREEN_WIDHT  = 400
SCREEN_HEIGHT = 600
SPEED         = 20
GRAVITY       = 2.5
GAME_SPEED    = 15

GROUND_WIDHT  = 2 * SCREEN_WIDHT
GROUND_HEIGHT = 100

PIPE_WIDHT = 80
PIPE_HEIGHT = 500
PIPE_GAP    = 300

wing = 'assets/audio/wing.wav'
hit  = 'assets/audio/hit.wav'

pygame.mixer.init()

# ─── EYE BLINK DETECTION (mediapipe Tasks API) ───────────────────────────────
eye_gesture = {"flap": False}

# Điểm Face Mesh 478 (tasks API có refine_landmarks):
# Mắt trái (từ góc nhìn camera): top=159, bottom=145, left=33, right=133
# Mắt phải: top=386, bottom=374, left=362, right=263
LEFT_EYE  = dict(top=159, bottom=145, left=33, right=133)
RIGHT_EYE = dict(top=386, bottom=374, left=362, right=263)

EAR_THRESHOLD = 0.20
BLINK_CONSEC  = 2

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')

def compute_ear(lm, eye):
    v = abs(lm[eye['top']].y - lm[eye['bottom']].y)
    h = abs(lm[eye['left']].x - lm[eye['right']].x)
    return v / h if h > 0 else 0.0

def eye_detection_thread():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = mp_vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    blink_counter  = 0
    blink_triggered = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        h, w = frame.shape[:2]
        status_text = "Khong thay mat..."
        color = (0, 0, 255)

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            left_ear  = compute_ear(lm, LEFT_EYE)
            right_ear = compute_ear(lm, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                blink_counter += 1
                color = (0, 140, 255)
                status_text = f"NHEO MAT... (EAR={ear:.2f})"
            else:
                if blink_counter >= BLINK_CONSEC and not blink_triggered:
                    eye_gesture["flap"] = True
                    blink_triggered = True
                else:
                    blink_triggered = False
                blink_counter = 0
                color = (0, 220, 0)
                status_text = f"Mat mo (EAR={ear:.2f})"

            # Vẽ điểm mắt
            for key in LEFT_EYE.values():
                px = int(lm[key].x * w)
                py = int(lm[key].y * h)
                cv2.circle(frame, (px, py), 3, (255, 255, 0), -1)
            for key in RIGHT_EYE.values():
                px = int(lm[key].x * w)
                py = int(lm[key].y * h)
                cv2.circle(frame, (px, py), 3, (255, 255, 0), -1)

        cv2.putText(frame, status_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "Nhay mat de chim bay | Q: thoat",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Eye Blink Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

threading.Thread(target=eye_detection_thread, daemon=True).start()

# ─── SPRITE CLASSES ───────────────────────────────────────────────────────────
class Bird(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.images = [
            pygame.image.load('assets/sprites/redbird-upflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/redbird-midflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/redbird-downflap.png').convert_alpha()
        ]
        self.speed = SPEED
        self.current_image = 0
        self.image = self.images[0]
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDHT / 6
        self.rect[1] = SCREEN_HEIGHT / 2

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.speed += GRAVITY
        self.rect[1] += self.speed

    def bump(self):
        self.speed = -SPEED

    def begin(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]


class Pipe(pygame.sprite.Sprite):
    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/pipe-red.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDHT, PIPE_HEIGHT))
        self.rect  = self.image.get_rect()
        self.rect[0] = xpos
        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = -(self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect[0] -= GAME_SPEED


class Ground(pygame.sprite.Sprite):
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDHT, GROUND_HEIGHT))
        self.mask  = pygame.mask.from_surface(self.image)
        self.rect  = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED


def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])

def get_random_pipes(xpos):
    size = random.randint(100, 300)
    return Pipe(False, xpos, size), Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)

def do_flap(bird):
    bird.bump()
    try:
        pygame.mixer.music.load(wing)
        pygame.mixer.music.play()
    except:
        pass

# ─── PYGAME INIT ──────────────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird - Eye Blink Control')

BACKGROUND  = pygame.image.load('assets/sprites/background-night.png')
BACKGROUND  = pygame.transform.scale(BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))
BEGIN_IMAGE = pygame.image.load('assets/sprites/message.png').convert_alpha()

bird_group = pygame.sprite.Group()
bird = Bird()
bird_group.add(bird)

ground_group = pygame.sprite.Group()
for i in range(2):
    ground_group.add(Ground(GROUND_WIDHT * i))

pipe_group = pygame.sprite.Group()
for i in range(2):
    pipes = get_random_pipes(SCREEN_WIDHT * i + 800)
    pipe_group.add(pipes[0])
    pipe_group.add(pipes[1])

clock = pygame.time.Clock()
font  = pygame.font.SysFont('Arial', 15)

# ─── BEGIN SCREEN ─────────────────────────────────────────────────────────────
begin = True
while begin:
    clock.tick(15)
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit(); exit()
        if event.type == KEYDOWN:
            if event.key == K_SPACE or event.key == K_UP:
                do_flap(bird); begin = False

    if eye_gesture["flap"]:
        eye_gesture["flap"] = False
        do_flap(bird); begin = False

    screen.blit(BACKGROUND, (0, 0))
    screen.blit(BEGIN_IMAGE, (120, 150))
    hint = font.render("Space hoac Nhay mat de bat dau", True, (255, 255, 0))
    screen.blit(hint, (30, 10))

    if is_off_screen(ground_group.sprites()[0]):
        ground_group.remove(ground_group.sprites()[0])
        ground_group.add(Ground(GROUND_WIDHT - 20))

    bird.begin()
    ground_group.update()
    bird_group.draw(screen)
    ground_group.draw(screen)
    pygame.display.update()

# ─── MAIN GAME LOOP ───────────────────────────────────────────────────────────
while True:
    clock.tick(15)
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit(); exit()
        if event.type == KEYDOWN:
            if event.key == K_SPACE or event.key == K_UP:
                do_flap(bird)

    if eye_gesture["flap"]:
        eye_gesture["flap"] = False
        do_flap(bird)

    screen.blit(BACKGROUND, (0, 0))

    if is_off_screen(ground_group.sprites()[0]):
        ground_group.remove(ground_group.sprites()[0])
        ground_group.add(Ground(GROUND_WIDHT - 20))

    if is_off_screen(pipe_group.sprites()[0]):
        pipe_group.remove(pipe_group.sprites()[0])
        pipe_group.remove(pipe_group.sprites()[0])
        pipes = get_random_pipes(SCREEN_WIDHT * 2)
        pipe_group.add(pipes[0])
        pipe_group.add(pipes[1])

    bird_group.update()
    ground_group.update()
    pipe_group.update()

    bird_group.draw(screen)
    pipe_group.draw(screen)
    ground_group.draw(screen)

    hint = font.render("Nhay mat de bay | Space du phong", True, (255, 255, 0))
    screen.blit(hint, (30, 10))
    pygame.display.update()

    if (pygame.sprite.groupcollide(bird_group, ground_group, False, False, pygame.sprite.collide_mask) or
            pygame.sprite.groupcollide(bird_group, pipe_group, False, False, pygame.sprite.collide_mask)):
        try:
            pygame.mixer.music.load(hit)
            pygame.mixer.music.play()
        except Exception:
            pass
        time.sleep(1)
        break

pygame.quit()
try:
    sys.exit(0)
except SystemExit:
    os._exit(0)
