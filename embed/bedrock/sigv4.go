package bedrock

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"
)

const (
	algorithm = "AWS4-HMAC-SHA256"
	service   = "bedrock"
)

// signRequest applies AWS Signature Version 4 to the given HTTP request.
// The body must already be set and bodyHash must be the hex-encoded SHA-256
// of the request body.
func signRequest(req *http.Request, accessKey, secretKey, region, bodyHash string, t time.Time) {
	datestamp := t.UTC().Format("20060102")
	amzDate := t.UTC().Format("20060102T150405Z")
	credentialScope := fmt.Sprintf("%s/%s/%s/aws4_request", datestamp, region, service)

	// Set required headers before signing.
	req.Header.Set("x-amz-date", amzDate)
	req.Header.Set("x-amz-content-sha256", bodyHash)

	// Build canonical headers and signed headers list.
	signedHeaderKeys := []string{"content-type", "host", "x-amz-content-sha256", "x-amz-date"}
	sort.Strings(signedHeaderKeys)
	signedHeaders := strings.Join(signedHeaderKeys, ";")

	var canonicalHeaders strings.Builder
	for _, key := range signedHeaderKeys {
		var val string
		switch key {
		case "host":
			val = req.Host
			if val == "" {
				val = req.URL.Host
			}
		default:
			val = req.Header.Get(key)
		}
		canonicalHeaders.WriteString(key)
		canonicalHeaders.WriteByte(':')
		canonicalHeaders.WriteString(strings.TrimSpace(val))
		canonicalHeaders.WriteByte('\n')
	}

	// Build canonical request.
	canonicalURI := req.URL.Path
	if canonicalURI == "" {
		canonicalURI = "/"
	}
	canonicalQueryString := req.URL.RawQuery

	canonicalRequest := strings.Join([]string{
		req.Method,
		canonicalURI,
		canonicalQueryString,
		canonicalHeaders.String(),
		signedHeaders,
		bodyHash,
	}, "\n")

	// Create string to sign.
	canonicalRequestHash := sha256Hex([]byte(canonicalRequest))
	stringToSign := strings.Join([]string{
		algorithm,
		amzDate,
		credentialScope,
		canonicalRequestHash,
	}, "\n")

	// Calculate signature.
	signingKey := deriveSigningKey(secretKey, datestamp, region, service)
	signature := hex.EncodeToString(hmacSHA256(signingKey, []byte(stringToSign)))

	// Set Authorization header.
	authHeader := fmt.Sprintf("%s Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		algorithm, accessKey, credentialScope, signedHeaders, signature)
	req.Header.Set("Authorization", authHeader)
}

// deriveSigningKey derives the AWS SigV4 signing key.
func deriveSigningKey(secretKey, datestamp, region, svc string) []byte {
	kDate := hmacSHA256([]byte("AWS4"+secretKey), []byte(datestamp))
	kRegion := hmacSHA256(kDate, []byte(region))
	kService := hmacSHA256(kRegion, []byte(svc))
	kSigning := hmacSHA256(kService, []byte("aws4_request"))
	return kSigning
}

// hmacSHA256 returns the HMAC-SHA256 of data using the given key.
func hmacSHA256(key, data []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(data)
	return h.Sum(nil)
}

// sha256Hex returns the hex-encoded SHA-256 hash of data.
func sha256Hex(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}
