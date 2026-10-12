import type {
  ComplaintCreateRequest,
  ComplaintCreateResponse,
  ComplaintListItem,
  ComplaintReviewRequest,
  ComplaintReviewResponse,
  ComplaintPageResponse,
  ComplaintStatus,
} from "../types/complaint";

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


async function getErrorMessage(
  response: Response,
): Promise<string> {
  try {
    const body =
      await response.json();

    if (
      typeof body?.detail
      === "string"
    ) {
      return body.detail;
    }

    if (
      Array.isArray(
        body?.detail,
      )
      && body.detail.length > 0
    ) {
      const firstError =
        body.detail[0];

      if (
        typeof firstError?.msg
        === "string"
      ) {
        return firstError.msg;
      }
    }
  } catch {
    // JSON olmayan hata cevabında
    // genel mesaj kullanılır.
  }

  return (
    "İstek gerçekleştirilemedi. "
    + `HTTP ${response.status}`
  );
}


export async function createComplaint(
  request: ComplaintCreateRequest,
): Promise<ComplaintCreateResponse> {
  const response = await fetch(
    `${getApiBaseUrl()}/complaints`,
    {
      method: "POST",

      headers: {
        "Content-Type":
          "application/json",
      },

      body: JSON.stringify(
        request,
      ),
    },
  );

  if (!response.ok) {
    throw new Error(
      await getErrorMessage(
        response,
      ),
    );
  }

  return response.json();
}


export async function getReviewQueue(): Promise<
  ComplaintListItem[]
> {
  const accessToken =
    sessionStorage.getItem(
      "access_token",
    );

  if (!accessToken) {
    throw new Error(
      "Oturum bulunamadı.",
    );
  }

  const response = await fetch(
    (
      `${getApiBaseUrl()}`
      + "/complaints"
      + "?review_required=true"
    ),
    {
      headers: {
        Authorization:
          `Bearer ${accessToken}`,
      },
    },
  );

  if (
    response.status === 401
  ) {
    sessionStorage.clear();

    throw new Error(
      "Oturumunuz sona erdi.",
    );
  }

  if (!response.ok) {
    throw new Error(
      await getErrorMessage(
        response,
      ),
    );
  }

  return response.json();
}
export async function reviewComplaint(
  complaintId: number,
  request: ComplaintReviewRequest,
): Promise<ComplaintReviewResponse> {
  const accessToken =
    sessionStorage.getItem(
      "access_token",
    );

  if (!accessToken) {
    throw new Error(
      "Oturum bulunamadı.",
    );
  }

  const response = await fetch(
    (
      `${getApiBaseUrl()}`
      + `/complaints/${complaintId}/review`
    ),
    {
      method: "POST",

      headers: {
        "Content-Type":
          "application/json",

        Authorization:
          `Bearer ${accessToken}`,
      },

      body: JSON.stringify(
        request,
      ),
    },
  );

  if (
    response.status === 401
  ) {
    sessionStorage.clear();

    throw new Error(
      "Oturumunuz sona erdi.",
    );
  }

  if (!response.ok) {
    throw new Error(
      await getErrorMessage(
        response,
      ),
    );
  }

  return response.json();
}
export async function getComplaintsByStatus(
  status: ComplaintStatus,
  page = 1,
  pageSize = 100,
): Promise<ComplaintPageResponse> {
  const accessToken =
    sessionStorage.getItem(
      "access_token",
    );

  if (!accessToken) {
    throw new Error(
      "Oturum bulunamadı.",
    );
  }

  if (!API_BASE_URL) {
    throw new Error(
      "API adresi yapılandırılmamış.",
    );
  }

  const params =
    new URLSearchParams({
      status,
      page: String(page),
      page_size: String(pageSize),
    });

  const response = await fetch(
    (
      `${API_BASE_URL}/complaints/paged?`
      + params.toString()
    ),
    {
      headers: {
        Authorization:
          `Bearer ${accessToken}`,
      },
    },
  );

  if (response.status === 401) {
    sessionStorage.clear();

    throw new Error(
      "Oturumunuz sona erdi.",
    );
  }

  if (response.status === 403) {
    throw new Error(
      "Bu işlem için yetkiniz yok.",
    );
  }

  if (!response.ok) {
    let message =
      "Başvurular yüklenemedi.";

    try {
      const body =
        await response.json();

      if (
        body
        && typeof body.detail
        === "string"
      ) {
        message =
          body.detail;
      }
    } catch {
      // Genel hata mesajı kullanılır.
    }

    throw new Error(
      message,
    );
  }

  return response.json();
}
export async function getComplaintByApplicationNo(
  applicationNo: string,
): Promise<ComplaintListItem> {
  const accessToken =
    sessionStorage.getItem(
      "access_token",
    );

  if (!accessToken) {
    throw new Error(
      "Oturum bulunamadı.",
    );
  }

  const normalizedApplicationNo =
    applicationNo
      .trim()
      .toUpperCase();

  const response = await fetch(
    (
      `${getApiBaseUrl()}`
      + "/complaints/by-application-no/"
      + encodeURIComponent(
        normalizedApplicationNo,
      )
    ),
    {
      headers: {
        Authorization:
          `Bearer ${accessToken}`,
      },
    },
  );

  if (
    response.status === 401
  ) {
    sessionStorage.clear();

    throw new Error(
      "Oturumunuz sona erdi.",
    );
  }

  if (
    response.status === 404
  ) {
    throw new Error(
      "Bu başvuru numarasına ait kayıt bulunamadı.",
    );
  }

  if (!response.ok) {
    throw new Error(
      await getErrorMessage(
        response,
      ),
    );
  }

  return response.json();
}