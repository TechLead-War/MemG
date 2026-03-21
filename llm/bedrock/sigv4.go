package bedrock

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

const (
	algorithm = "AWS4-HMAC-SHA256"
	service   = "bedrock"
)

// signRequest adds AWS SigV4 headers to the given HTTP request.
func signRequest(req *http.Request, payloadHash string, accessKey, secretKey, region string, t time.Time) {
	datestamp := t.UTC().Format("20060102")
	amzDate := t.UTC().Format("20060102T150405Z")
	scope := fmt.Sprintf("%s/%s/%s/aws4_request", datestamp, region, service)

	req.Header.Set("x-amz-date", amzDate)
	req.Header.Set("x-amz-content-sha256", payloadHash)

	// 1. Create canonical request.
	canonicalURI := req.URL.Path
	if canonicalURI == "" {
		canonicalURI = "/"
	}
	canonicalQueryString := canonicalQuery(req.URL.Query())

	signedHeaderNames := signedHeaders(req)
	canonicalHeaders := canonicalHeaderString(req, signedHeaderNames)

	canonicalRequest := strings.Join([]string{
		req.Method,
		canonicalURI,
		canonicalQueryString,
		canonicalHeaders,
		strings.Join(signedHeaderNames, ";"),
		payloadHash,
	}, "\n")

	// 2. Create string to sign.
	canonicalRequestHash := hashSHA256([]byte(canonicalRequest))
	stringToSign := strings.Join([]string{
		algorithm,
		amzDate,
		scope,
		canonicalRequestHash,
	}, "\n")

	// 3. Derive signing key.
	signingKey := deriveSigningKey(secretKey, datestamp, region, service)

	// 4. Calculate signature.
	signature := hex.EncodeToString(hmacSHA256(signingKey, []byte(stringToSign)))

	// 5. Add authorization header.
	authHeader := fmt.Sprintf("%s Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		algorithm, accessKey, scope, strings.Join(signedHeaderNames, ";"), signature)
	req.Header.Set("Authorization", authHeader)
}

func deriveSigningKey(secret, datestamp, region, svc string) []byte {
	kDate := hmacSHA256([]byte("AWS4"+secret), []byte(datestamp))
	kRegion := hmacSHA256(kDate, []byte(region))
	kService := hmacSHA256(kRegion, []byte(svc))
	kSigning := hmacSHA256(kService, []byte("aws4_request"))
	return kSigning
}

func hmacSHA256(key, data []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(data)
	return h.Sum(nil)
}

func hashSHA256(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

func signedHeaders(req *http.Request) []string {
	var names []string
	for name := range req.Header {
		names = append(names, strings.ToLower(name))
	}
	// Always include host.
	hasHost := false
	for _, n := range names {
		if n == "host" {
			hasHost = true
			break
		}
	}
	if !hasHost {
		names = append(names, "host")
	}
	sort.Strings(names)
	return names
}

func canonicalHeaderString(req *http.Request, sorted []string) string {
	var b strings.Builder
	for _, name := range sorted {
		if name == "host" {
			b.WriteString("host:")
			b.WriteString(req.URL.Host)
			b.WriteByte('\n')
		} else {
			values := req.Header.Values(http.CanonicalHeaderKey(name))
			b.WriteString(name)
			b.WriteByte(':')
			b.WriteString(strings.Join(values, ","))
			b.WriteByte('\n')
		}
	}
	return b.String()
}

func canonicalQuery(values url.Values) string {
	if len(values) == 0 {
		return ""
	}
	keys := make([]string, 0, len(values))
	for k := range values {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var parts []string
	for _, k := range keys {
		vs := values[k]
		sort.Strings(vs)
		for _, v := range vs {
			parts = append(parts, url.QueryEscape(k)+"="+url.QueryEscape(v))
		}
	}
	return strings.Join(parts, "&")
}
