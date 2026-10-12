export interface ComplaintCreateRequest {
  complaint_text: string;
  neighborhood: string;
  street_name: string | null;
  address_text: string | null;
  latitude: number;
  longitude: number;
}


export interface ComplaintCreateResponse {
  id: number;
  application_no: string;

  neighborhood: string | null;
  street_name: string | null;
  address_text: string | null;

  predicted_category: string;
  secondary_category: string;

  model_score: number;
  second_model_score: number;
  score_margin: number;

  review_required: boolean;
  review_reason: string | null;

  status: string;
  model_version: string;

  latitude: number;
  longitude: number;

  created_at: string;
}
export interface ComplaintListItem {
  id: number;
  application_no: string;
  complaint_text: string;

  neighborhood: string | null;
  street_name: string | null;
  address_text: string | null;

  predicted_category: string;
  secondary_category: string;

  model_score: number;
  score_margin: number;

  review_required: boolean;
  review_reason: string | null;

  final_category: string | null;
  was_corrected: boolean | null;

  status: string;

  latitude: number;
  longitude: number;

  created_at: string;
}
export interface ComplaintPageResponse {
  items: ComplaintListItem[];

  total: number;

  page: number;
  page_size: number;
  total_pages: number;
}
export type ComplaintCategory =
  | "TEMIZLIK_ATIK"
  | "YOL_KALDIRIM"
  | "PARK_YESIL_ALAN"
  | "KENT_DUZENI_DENETIM"
  | "HAYVAN_VETERINER";


export interface ComplaintReviewRequest {
  final_category: ComplaintCategory;
}


export interface ComplaintReviewResponse {
  id: number;
  application_no: string;

  predicted_category: string;
  final_category: string;

  was_corrected: boolean;

  reviewed_by: number;
  reviewed_at: string;

  review_required: boolean;
  status: string;
}
export type ComplaintStatus =
  | "REVIEW_REQUIRED"
  | "AUTO_ACCEPTED"
  | "REVIEWED";