export interface AnalyticsSummary {
  total_complaints: number;

  auto_accepted: number;
  review_pending: number;
  reviewed: number;

  human_review_total: number;
  corrected_by_human: number;

  auto_acceptance_rate: number;
  human_review_rate: number;
  correction_rate: number;
}


export interface CategoryDistributionItem {
  category: string;
  count: number;
  percentage: number;
}


export interface TrendItem {
  date: string;
  count: number;
}


export interface MapPointItem {
  id: number;
  application_no: string;

  complaint_text: string;
  neighborhood: string | null;

  category: string;
  status: string;

  latitude: number;
  longitude: number;

  created_at: string;
}


export interface NearbyComplaintItem {
  id: number;
  application_no: string;
  complaint_text: string;

  category: string;
  status: string;

  latitude: number;
  longitude: number;
  distance_meters: number;
}


export interface NearbyComplaintsResponse {
  center_latitude: number;
  center_longitude: number;

  radius_meters: number;
  category_filter: string | null;

  count: number;
  results: NearbyComplaintItem[];
}


export interface NeighborhoodAnalyticsItem {
  neighborhood: string;

  count: number;
  percentage: number;

  dominant_category: string;
  dominant_category_count: number;

  category_counts: Record<
    string,
    number
  >;
}



export interface ModelHumanCategoryItem {
  category: string;

  reviewed: number;
  confirmed: number;
  corrected: number;

  correction_rate: number;
}


export interface CorrectionFlowItem {
  predicted_category: string;
  final_category: string;
  count: number;
}


export interface ModelHumanComparisonResponse {
  reviewed: number;

  confirmed: number;
  corrected: number;

  agreement_rate: number;
  correction_rate: number;

  by_predicted_category:
    ModelHumanCategoryItem[];

  correction_flows:
    CorrectionFlowItem[];
}


export interface AnalyticsDateRange {
  fromDate?: string;
  toDate?: string;
}
