"""AI Enquiry Understanding capability.

Converts a free-text travel request into structured enquiry fields
plus a list of clarifying questions for fields the user did not
mention.

Read-only. Does not create enquiries, call vendors, or touch bookings.

Public surface (for the route):
    service.understand(message, ...) -> EnquiryUnderstandResponse
    schemas.EnquiryExtractRequest, EnquiryUnderstandResponse
    prompts.ENQUIRY_SYSTEM_PROMPT, ENQUIRY_PROMPT_VERSION
"""
