import streamlit as st
from PIL import Image, UnidentifiedImageError
import io, os, random, smtplib, requests, time
from email.mime.text import MIMEText
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import speech_recognition as sr
import cv2
from deepface import DeepFace
from fpdf import FPDF
from diffusers import StableDiffusionPipeline
import torch
import numpy as np

for key in [
    "face_verified", "voice_verified", "access_granted", "otp_sent", "otp_verified",
    "generated_key", "pdf_data", "encrypt_email", "otp_code", "otp_sent_time"
]:
    if key not in st.session_state:
        if key in ["face_verified", "voice_verified", "access_granted", "otp_sent", "otp_verified"]:
            st.session_state[key] = False
        else:
            st.session_state[key] = None


st.set_page_config(page_title="Secure Scripts AI System", layout="centered", page_icon="🔐")
st.title("🔐 Secure Scripts AI System")


TOGETHER_API_KEY = "16fe845937edfc5513d5a3adca83f7d53f207fe64aa32da3672e2fa224fa0579"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Content-Type": "application/json"
}


FACE_DB = "face_db"
os.makedirs(FACE_DB, exist_ok=True)

def generate_document(prompt: str, model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1") -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You generate clean, professional documents from prompts."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.7
    }
    try:
        response = requests.post(TOGETHER_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠ Error: {e}"

def text_to_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    font_path = "DejaVuSans.ttf"
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)
    else:
        pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    return pdf.output(dest='S').encode("latin1", errors="ignore")


def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def save_face_image(name, img):
    user_dir = os.path.join(FACE_DB, name)
    os.makedirs(user_dir, exist_ok=True)
    count = len(os.listdir(user_dir))
    img.save(os.path.join(user_dir, f"{count}.jpg"))


if not os.path.exists("face_db"):
    os.makedirs("face_db")

st.set_page_config(page_title="Secure Face Auth", layout="centered")
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb_img

def save_face(name, image):
    path = os.path.join("face_db", f"{name}.jpg")
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def verify_face(image):
    for filename in os.listdir("face_db"):
        if filename.endswith(".jpg"):
            path = os.path.join("face_db", filename)
            try:
                result = DeepFace.verify(
                    img1_path=image,
                    img2_path=path,
                    model_name='VGG-Face',
                    detector_backend='opencv',
                    enforce_detection=True
                )
                if result["verified"] and result["distance"] < 0.35:  # More strict matching
                    return True, os.path.splitext(filename)[0]
            except Exception as e:
                print(f"Verification failed with {filename}: {e}")
    return False, None


def verify_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙 Say: 'your secret phrase'")
        try:
            audio = recognizer.listen(source, timeout=5)
            return "unlock secure system" in recognizer.recognize_google(audio).lower()
        except:
            return False


def generate_aes_key():
    return get_random_bytes(16)

def encrypt_pdf(data: bytes, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC)
    return cipher.iv + cipher.encrypt(pad(data, AES.block_size))

def decrypt_pdf(encrypted_data: bytes, key: bytes) -> bytes:
    iv = encrypted_data[:16]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(encrypted_data[16:]), AES.block_size)

def embed_key_in_image(image_bytes, aes_key):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        st.error("Not a valid image.")
        return None
    key_bin = ''.join(format(byte, '08b') for byte in aes_key) + '11111110'
    pixels, new_pixels, idx = list(img.getdata()), [], 0
    for pixel in pixels:
        r, g, b = pixel
        if idx < len(key_bin): r = (r & ~1) | int(key_bin[idx]); idx += 1
        if idx < len(key_bin): g = (g & ~1) | int(key_bin[idx]); idx += 1
        if idx < len(key_bin): b = (b & ~1) | int(key_bin[idx]); idx += 1
        new_pixels.append((r, g, b))
    img.putdata(new_pixels)
    out = io.BytesIO(); img.save(out, format="PNG")
    return out.getvalue()

def extract_key_from_image(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError:
        st.error("Invalid image.")
        return b""
    binary = ''.join(str(value & 1) for pixel in img.getdata() for value in pixel)
    byte_list = [binary[i:i+8] for i in range(0, len(binary), 8)]
    result = bytearray()
    for b in byte_list:
        if b == '11111110': break
        result.append(int(b, 2))
    return bytes(result)


def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp_via_email(recipient, otp, decryptor_email=None):
    sender = "nagalakshmipala06@gmail.com"
    password = "pcccxlorrrzoyeif"
    body = f"🔐 Your OTP is: {otp}\n\n Decryption attempt by:\n{decryptor_email}" if decryptor_email else f"🔐 Your OTP is: {otp}"
    message = MIMEText(body)
    message['Subject'] = "OTP for SecureScript Decryption"
    message['From'] = sender
    message['To'] = recipient
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.send_message(message)

if not st.session_state["face_verified"] or not st.session_state["voice_verified"]:
    
   st.subheader(" Register Face")
name = st.text_input("Enter Name for Registration")

if st.button("Register Face"):
    if name.strip() == "":
        st.warning(" Please enter a valid name.")
    else:
        image = capture_image()
        if image is not None:
            save_face(name, image)
            st.image(image, width=150, caption="Registered Face")
            st.success(f" Face registered for {name}")
        else:
            st.error("Could not capture image. Try again.")


st.subheader("Verify Face")

if st.button("Verify Face"):
    image = capture_image()
    if image is not None:
        st.image(image, width=150, caption="Face to Verify")
       
        temp_path = "temp_verify.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        matched, person = verify_face(temp_path)
        if matched:
            st.success(f"Face verified! Welcome, {person}")
            st.session_state["face_verified"] = True
        else:
            st.error("Face not recognized.")
        os.remove(temp_path)
    else:
        st.error(" Could not capture image. Try again.")

    st.subheader(" Voice Access")
if st.button("Verify Voice"):
    if verify_voice():
        st.session_state["voice_verified"] = True
        st.success(" Voice verified!")
    else:
        st.error(" Voice not recognized.")


if st.session_state["face_verified"] and st.session_state["voice_verified"]:
    st.session_state["access_granted"] = True

is_authenticated = st.session_state["access_granted"]


if is_authenticated:
    st.success("Access Granted.")
    menu = st.sidebar.selectbox("Choose Action", [
        "📄 Generate Document", "📂 Upload PDF", "🌚 Encrypt PDF",
        "🕛 Stego Image", "🕵️ Extract AES Key", "🗂️ Decrypt PDF"])

    if menu == "📄 Generate Document":
        prompt = st.text_area("Enter prompt")
        if st.button("Generate") and prompt:
            content = generate_document(prompt)
            st.session_state.pdf_data = text_to_pdf(content)
            st.session_state.generated_key = generate_aes_key()
        if st.session_state.pdf_data:
            st.download_button(" Download PDF", st.session_state.pdf_data, "doc.pdf")
        if st.session_state.generated_key:
            st.download_button("🔑 Download Key", st.session_state.generated_key, "aes.key")

    elif menu == "📂 Upload PDF":
        file = st.file_uploader("Upload PDF")
        if file:
            key = generate_aes_key()
            st.download_button("🔑 Download Key", key, "aes.key")

    elif menu == "🌚 Encrypt PDF":
        email = st.text_input("Sender Email")
        pdf = st.file_uploader("Upload PDF")
        key = st.file_uploader("Upload AES Key")
        if email: st.session_state.encrypt_email = email
        if pdf and key:
            encrypted = encrypt_pdf(pdf.read(), key.read())
            st.download_button("Download Encrypted PDF", encrypted, "enc.enc")

    elif menu == "🕛 Stego Image":
        opt = st.radio("Choose", ["Upload", "Generate"])
        key_file = st.file_uploader("AES Key")
        if opt == "Upload":
            img = st.file_uploader("Upload Image")
            if img and key_file:
                result = embed_key_in_image(img.read(), key_file.read())
                if result:
                    st.download_button("Download Stego Image", result, "stego.png")
        elif opt == "Generate":
            prompt = st.text_input("AI Image Prompt")
            if st.button("Generate AI Image") and prompt and key_file:
                pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
                ).to("cuda")
                image = pipe(prompt).images[0]
                img_bytes = io.BytesIO(); image.save(img_bytes, "PNG")
                stego = embed_key_in_image(img_bytes.getvalue(), key_file.read())
                st.image(image)
                if stego:
                    st.download_button("Download Stego Image", stego, "ai_stego.png")

    elif menu == "🕵️ Extract AES Key":
        img = st.file_uploader("Stego Image")
        if img:
            key = extract_key_from_image(img.read())
            st.download_button("Extracted Key", key, "extracted.key")

    elif menu == "🗂️ Decrypt PDF":
        decryptor_email = st.text_input("Your Email")
        current_time = time.time()
        if st.session_state.otp_sent_time is None or current_time - st.session_state.otp_sent_time > 120:
            if st.button("Send OTP to Sender"):
                if not decryptor_email:
                    st.error("Enter email.")
                elif not st.session_state.encrypt_email:
                    st.error("No sender email found.")
                else:
                    otp = generate_otp()
                    st.session_state.otp_code = otp
                    st.session_state.otp_sent_time = current_time
                    send_otp_via_email(st.session_state.encrypt_email, otp, decryptor_email)
                    st.success(f"OTP sent to sender: {st.session_state.encrypt_email}")
        else:
            left = 120 - int(current_time - st.session_state.otp_sent_time)
            st.info(f"Wait {left} sec before resending OTP.")

        user_otp = st.text_input("Enter OTP", type="password")
        if st.button("Verify OTP") and user_otp == st.session_state.otp_code:
            st.session_state.otp_verified = True
            st.success("OTP Verified.")
        elif user_otp:
            st.error("Incorrect OTP.")

        if st.session_state.otp_verified:
            enc = st.file_uploader("Encrypted File", type=["enc"])
            key = st.file_uploader("AES Key", type=["key"])
            if enc and key:
                try:
                    decrypted = decrypt_pdf(enc.read(), key.read())
                    st.download_button("Download Decrypted PDF", decrypted, "decrypted.pdf")
                except Exception as e:
                    st.error(f"Decryption failed: {e}")
else:
    st.info("🔐 Authenticate using Face and Voice to proceed.")
