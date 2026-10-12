import type {
  HealthResponse,
} from "../types/health";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL;

export async function getHealth(): Promise<HealthResponse> {
  if (!API_BASE_URL) {
    throw new Error(
      "VITE_API_BASE_URL is not configured.",
    );
  }

  const response = await fetch(
    `${API_BASE_URL}/health`,
  );

  if (!response.ok) {
    throw new Error(
      `Health request failed: ${response.status}`,
    );
  }

  return response.json();
}