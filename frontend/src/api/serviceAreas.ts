import type {
  ServiceAreaFeature,
} from "../types/serviceArea";


const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL;


export async function getSaricamServiceArea():
Promise<ServiceAreaFeature> {
  if (!API_BASE_URL) {
    throw new Error(
      "VITE_API_BASE_URL is not configured.",
    );
  }

  const response = await fetch(
    `${API_BASE_URL}/service-areas/SARICAM`,
  );

  if (!response.ok) {
    throw new Error(
      `Service area request failed: ${response.status}`,
    );
  }

  return response.json();
}