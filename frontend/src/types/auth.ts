export interface LoginRequest {
  username: string;
  password: string;
}


export interface LoginResponse {
  access_token: string;
  token_type: string;

  user_id: number;
  username: string;
  role: "OPERATOR" | "MANAGER";
}