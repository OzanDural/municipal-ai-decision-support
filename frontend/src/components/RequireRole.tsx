import {
  Navigate,
} from "react-router-dom";

import type {
  ReactNode,
} from "react";


interface RequireRoleProps {
  role:
    | "OPERATOR"
    | "MANAGER";

  children: ReactNode;
}


function RequireRole({
  role,
  children,
}: RequireRoleProps) {
  const accessToken =
    sessionStorage.getItem(
      "access_token",
    );

  const userRole =
    sessionStorage.getItem(
      "user_role",
    );

  if (!accessToken) {
    return (
      <Navigate
        to="/login"
        replace
      />
    );
  }

  if (userRole !== role) {
    if (
      userRole === "OPERATOR"
    ) {
      return (
        <Navigate
          to="/operator"
          replace
        />
      );
    }

    if (
      userRole === "MANAGER"
    ) {
      return (
        <Navigate
          to="/manager"
          replace
        />
      );
    }

    sessionStorage.clear();

    return (
      <Navigate
        to="/login"
        replace
      />
    );
  }

  return children;
}


export default RequireRole;