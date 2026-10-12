import type {
  LoginRequest,
  LoginResponse,
} from "../types/auth";


const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL;


function getApiBaseUrl(): string {
  if (!API_BASE_URL) {
    throw new Error(
      "VITE_API_BASE_URL is not configured.",
    );
  }

  return API_BASE_URL;
}


async function getErrorMessage(
  response: Response,
): Promise<string> {
  try {
    const body =
      await response.json();

    if (
      typeof body?.detail
      === "string"
    ) {
      return body.detail;
    }
  } catch {
    // JSON olmayan hata cevabında
    // genel mesaj kullanılır.
  }

  return (
    "Giriş işlemi başarısız oldu. "
    + `HTTP ${response.status}`
  );
}


export async function login(
  request: LoginRequest,
): Promise<LoginResponse> {
  const response = await fetch(
    `${getApiBaseUrl()}/auth/login`,
    {
      method: "POST",

      headers: {
        "Content-Type":
          "application/json",
      },

      body: JSON.stringify(
        request,
      ),
    },
  );

  if (!response.ok) {
    throw new Error(
      await getErrorMessage(
        response,
      ),
    );
  }

  return response.json();
}