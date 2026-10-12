export interface HealthResponse {
  status: string;

  database: {
    connected: boolean;
    user: string;
    name: string;
  };

  postgis: {
    available: boolean;
    version: string;
  };

  model: {
    loaded: boolean;
    device: string;
    version: string;
    load_seconds: number;
  };
}
