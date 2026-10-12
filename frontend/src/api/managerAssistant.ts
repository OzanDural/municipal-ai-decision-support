import type {
  ManagerAssistantRequest,
  ManagerAssistantResponse,
} from "../types/managerAssistant";


const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL
  ?? "http://127.0.0.1:8000";


interface ErrorResponse {
  detail?: string;
}


async function readErrorDetail(
  response: Response,
): Promise<string | null> {
  try {
    const body =
      await response.json() as ErrorResponse;

    if (
      typeof body.detail === "string"
      && body.detail.trim()
    ) {
      return body.detail.trim();
    }
  } catch {
    // Response body may be empty/non-JSON.
  }

  return null;
}


export async function askManagerAssistant(
  payload: ManagerAssistantRequest,
): Promise<ManagerAssistantResponse> {
  const token =
    sessionStorage.getItem(
      "access_token",
    );

  if (!token) {
    throw new Error(
      "Oturumunuz sona erdi.",
    );
  }

  const response = await fetch(
    `${API_BASE_URL}/manager-assistant/chat`,
    {
      method: "POST",
      headers: {
        "Content-Type":
          "application/json",
        Authorization:
          `Bearer ${token}`,
      },
      body: JSON.stringify(
        payload,
      ),
    },
  );

  if (response.status === 401) {
    sessionStorage.removeItem(
      "access_token",
    );

    sessionStorage.removeItem(
      "role",
    );

    sessionStorage.removeItem(
      "username",
    );

    throw new Error(
      "Oturumunuz sona erdi.",
    );
  }

  if (response.status === 403) {
    throw new Error(
      "Bu özellik yalnızca yönetici rolü için kullanılabilir.",
    );
  }

  if (response.status === 429) {
    const detail =
      await readErrorDetail(
        response,
      );

    throw new Error(
      detail
      ?? (
        "Gemini geçici istek sınırına ulaştı. "
        + "Kısa süre sonra tekrar deneyin."
      ),
    );
  }

  if (response.status === 503) {
    throw new Error(
      "Yönetici analitik asistanına şu anda ulaşılamıyor.",
    );
  }

  if (!response.ok) {
    const detail =
      await readErrorDetail(
        response,
      );

    throw new Error(
      detail
      ?? (
        "Yönetici analitik asistanı "
        + "isteği tamamlanamadı."
      ),
    );
  }

  return (
    await response.json()
  ) as ManagerAssistantResponse;
}
