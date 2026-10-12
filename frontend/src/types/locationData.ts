export interface NeighborhoodListResponse {
  district_code: string;
  district_name: string;
  count: number;
  neighborhoods: string[];
}


export interface StreetListResponse {
  district_code: string;
  district_name: string;

  source: string;
  retrieved_at: string;

  count: number;
  streets: string[];
}
export interface NeighborhoodCenter {
  name: string;
  latitude: number;
  longitude: number;
}

export interface NeighborhoodCenterListResponse {
  district_code: string;
  district_name: string;
  count: number;
  centers: NeighborhoodCenter[];
}