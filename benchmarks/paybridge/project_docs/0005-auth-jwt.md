# 0005. 인증 방식으로 JWT 채택

**Status:** Accepted
**Date:** 2026-01-15
**Owner:** 최민재 (플랫폼팀)

## Context
세션 기반 인증은 서버 확장 시 세션 스토어 동기화 문제가 있다.

## Decision
Stateless JWT 기반 인증으로 전환한다. Access Token 15분, Refresh Token 14일로 설정한다.

## Consequences
- 긍정적: 서버 수평 확장이 용이해짐
- 부정적: 토큰 즉시 폐기(revoke)가 어려워, 별도 블랙리스트 관리가 필요함
