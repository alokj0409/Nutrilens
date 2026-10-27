export async function startCamera(
  videoElement: HTMLVideoElement,
  facingMode: "environment" | "user" = "environment"
): Promise<MediaStream> {
  if (typeof navigator === "undefined" || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      "Camera is not supported or context is insecure. Access via http://localhost:8443 or HTTPS."
    );
  }

  let stream: MediaStream;
  try {
    // Attempt with requested ideal facingMode
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
  } catch {
    // Fallback: try any video source without strict facing constraints (essential on laptop/desktop webcams)
    stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
  }

  videoElement.srcObject = stream;
  videoElement.setAttribute("playsinline", "true");
  videoElement.muted = true;
  try {
    await videoElement.play();
  } catch (err) {
    console.warn("Video element play warning:", err);
  }
  return stream;
}

export function captureFrame(videoElement: HTMLVideoElement): string {
  const canvas = document.createElement("canvas");
  canvas.width = videoElement.videoWidth || 640;
  canvas.height = videoElement.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not get canvas context");
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.85);
}

export function stopCamera(stream: MediaStream | null): void {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
}

export function pickFromGallery(): Promise<string> {
  return new Promise((resolve, reject) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.style.display = "none";
    document.body.appendChild(input);

    const cleanup = () => input.remove();

    input.onchange = () => {
      const file = input.files?.[0];
      if (!file) {
        cleanup();
        reject(new Error("No file selected"));
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        cleanup();
        if (typeof reader.result === "string") {
          resolve(reader.result);
        } else {
          reject(new Error("Failed to read file"));
        }
      };
      reader.onerror = () => {
        cleanup();
        reject(new Error("Failed to read file"));
      };
      reader.readAsDataURL(file);
    };
    input.addEventListener(
      "cancel",
      () => {
        cleanup();
        reject(new Error("No file selected"));
      },
      { once: true },
    );
    input.click();
  });
}
