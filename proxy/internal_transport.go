package proxy

import "net/http"

// internalTransport wraps an http.RoundTripper and adds the X-MemG-Internal
// header to every request. This marks the request as originating from MemG
// itself (extraction, summarization, embedding) so the proxy skips
// interception and avoids infinite recursion.
type internalTransport struct {
	base http.RoundTripper
}

// NewInternalHTTPClient returns an *http.Client that adds the X-MemG-Internal
// header to every outgoing request.
func NewInternalHTTPClient() *http.Client {
	return &http.Client{
		Transport: &internalTransport{
			base: http.DefaultTransport,
		},
	}
}

func (t *internalTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.Header.Set(internalHeader, "true")
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	return base.RoundTrip(req)
}
