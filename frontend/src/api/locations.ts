import type {
  NeighborhoodCenterListResponse,
  NeighborhoodListResponse,
  StreetListResponse,
} from "../types/locationData";


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


export async function getNeighborhoods():
Promise<NeighborhoodListResponse> {
  const response = await fetch(
    `${getApiBaseUrl()}/locations/neighborhoods`,
  );

  if (!response.ok) {
    throw new Error(
      `Neighborhood request failed: ${response.status}`,
    );
  }

  return response.json();
}


export async function getStreets():
Promise<StreetListResponse> {
  const response = await fetch(
    `${getApiBaseUrl()}/locations/streets`,
  );

  if (!response.ok) {
    throw new Error(
      `Street request failed: ${response.status}`,
    );
  }

  return response.json();
}


export async function getNeighborhoodCenters():
Promise<NeighborhoodCenterListResponse> {
  const response = await fetch(
    `${getApiBaseUrl()}/locations/neighborhood-centers`,
  );

  if (!response.ok) {
    throw new Error(
      `Neighborhood center request failed: ${response.status}`,
    );
  }

  return response.json();
}