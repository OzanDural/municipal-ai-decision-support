export type ManagerAssistantAnalysisType =
  | "summary"
  | "category_distribution"
  | "trend"
  | "neighborhoods"
  | "model_human"
  | "executive_insight"
  | "nearby";


export type ManagerAssistantComplaintCategory =
  | "TEMIZLIK_ATIK"
  | "YOL_KALDIRIM"
  | "PARK_YESIL_ALAN"
  | "KENT_DUZENI_DENETIM"
  | "HAYVAN_VETERINER";


export interface ManagerAssistantRequest {
  message: string;

  analysis_type?:
    ManagerAssistantAnalysisType;

  from_date?: string;
  to_date?: string;

  latitude?: number;
  longitude?: number;
  radius_meters?: number;

  category?:
    ManagerAssistantComplaintCategory;
}


export interface ManagerAssistantResponse {
  answer: string;

  source:
    | "gemini"
    | "deterministic_fallback";

  notice?: string;
}
