import pygame, random, time, threading, cv2
import mediapipe as mp
from pygame.locals import *

# VARIABLES
SCREEN_WIDHT = 400
SCREEN_HEIGHT = 600
SPEED = 20
GRAVITY = 2.5
GAME_SPEED = 15

GROUND_WIDHT = 2 * SCREEN_WIDHT
GROUND_HEIGHT = 100

PIPE_WIDHT = 80
PIPE_HEIGHT = 500
PIPE_GAP = 300

wing = 'assets/audio/wing.wav'
hit  = 'assets/audio/hit.wav'

pygame.mixer.init()

# ─── HAND DETECTION ──────────────────────────────────────────────────────────
hand_gesture = {"flap": False}

def count_fingers(hand_landmarks):
    """Đếm số ngón tay đang giơ lên"""
    tips = [8, 12, 16, 20]   # đầu ngón trỏ, giữa, áp út, út
    pip  = [6, 10, 14, 18]   # khớp giữa tương ứng

    count = 0

    # 4 ngón (trừ ngón cái): đầu ngón cao hơn khớp giữa = đang giơ
    for t, p in zip(tips, pip):
        if hand_landmarks.landmark[t].y < hand_landmarks.landmark[p].y:
            count += 1

    # Ngón cái: so sánh trục X (tay phải: tip.x < ip.x)
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip  = hand_landmarks.landmark[3]
    if abs(thumb_tip.x - thumb_ip.x) > 0.04:
        count += 1

    return count

def hand_detection_thread():
    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    draw_utils = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        fingers = 0
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                draw_utils.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                fingers = count_fingers(lm)

            # Xoè 5 ngón → flap
            if fingers >= 5:
                hand_gesture["flap"] = True

            # Hiển thị số ngón + trạng thái
            color = (0, 255, 0) if fingers >= 5 else (0, 165, 255)
            label = f"Ngon tay: {fingers}  {'>>> BAY! <<<' if fingers >= 5 else ''}"
            cv2.putText(frame, label, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(frame, "Khong thay tay...", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, "Xoe 5 ngon de bay | Q: thoat", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow("Hand Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Khởi động thread camera
threading.Thread(target=hand_detection_thread, daemon=True).start()

# ─── SPRITE CLASSES ───────────────────────────────────────────────────────────
class Bird(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.images = [
            pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()
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
        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDHT, PIPE_HEIGHT))
        self.rect = self.image.get_rect()
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
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED


def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])

def get_random_pipes(xpos):
    size = random.randint(100, 300)
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted

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
pygame.display.set_caption('Flappy Bird - Hand Control')

BACKGROUND  = pygame.image.load('assets/sprites/background-day.png')
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
                do_flap(bird)
                begin = False

    if hand_gesture["flap"]:
        hand_gesture["flap"] = False
        do_flap(bird)
        begin = False

    screen.blit(BACKGROUND, (0, 0))
    screen.blit(BEGIN_IMAGE, (120, 150))

    hint = font.render("Space hoac Xoe 5 ngon de bat dau", True, (255, 255, 0))
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

    # Xoè 5 ngón → chim bay
    if hand_gesture["flap"]:
        hand_gesture["flap"] = False
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

    hint = font.render("Xoe 5 ngon de bay | Space du phong", True, (255, 255, 0))
    screen.blit(hint, (30, 10))

    pygame.display.update()

    if (pygame.sprite.groupcollide(bird_group, ground_group, False, False, pygame.sprite.collide_mask) or
            pygame.sprite.groupcollide(bird_group, pipe_group, False, False, pygame.sprite.collide_mask)):
        pygame.mixer.music.load(hit)
        pygame.mixer.music.play()
        time.sleep(1)
        break