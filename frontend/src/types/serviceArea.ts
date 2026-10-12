export interface ServiceAreaProperties {
  code: string;
  name: string;
  source: string;
}


export interface ServiceAreaGeometry {
  type: "MultiPolygon";

  coordinates: number[][][][];
}


export interface ServiceAreaFeature {
  type: "Feature";

  properties: ServiceAreaProperties;

  geometry: ServiceAreaGeometry;
}