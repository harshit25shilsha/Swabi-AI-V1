# SWABI KNOWLEDGE BASE SOURCE DOCUMENT

**Customer Support & Travel Knowledge**

- **Version:** 1.0
- **Audience:** RAG ingestion team, AI support developer, content ops
- **Source of truth:** Swabi BRD v1.1 (with Tracking Devices Amendment)
- **Status:** Ready for ingestion

---

## SECTION 0 — PURPOSE AND USAGE OF THIS DOCUMENT

This document is the single source for training and grounding the Swabi RAG-based customer support assistant. It contains nine top-level sections, each split into sub-sections that map cleanly to individual knowledge chunks for embedding.

### How the RAG will use this

Live marketplace data (package prices, vendor profiles, bids, bookings, wallet balances, escrow state, device tracking) is never answered from this document. Those answers are fetched from the Swabi backend APIs at query time.

This document is only for:

- Platform policies (how Swabi works, rules, contracts)
- Process flows (bidding, payment, escrow, cancellation, disputes, verification)
- Rules and formulas (fees, commissions, refund percentages, milestones)
- Reference knowledge (destinations, activities, travel tips)

### Chunking recommendation

Each numbered sub-section (for example, 1.3 Community Guidelines) is a single chunk. Do not split sub-sections unless they exceed 500 tokens. Do not merge sub-sections. Preserve headings verbatim in every chunk for retrieval precision.

### Metadata to attach to every chunk

| Field | Value |
|---|---|
| `source` | `Swabi Policy KB` |
| `section_id` | e.g., `1.3` |
| `section_title` | e.g., `Community Guidelines` |
| `category` | e.g., `Platform Policies` |
| `audience` | `customer` / `vendor` / `admin` / `all` |
| `last_updated` | `2026-10` |
| `version` | `1.0` |

---

## SECTION 1 — PLATFORM POLICIES

### 1.1 Terms of Service

**Who it applies to:** Customers, vendors, and anyone using Swabi.

**Acceptance:** By creating an account, posting an enquiry, placing a bid, or making a payment, the user agrees to these terms.

#### Customer obligations

- Provide accurate personal and trip information (traveler count, ages, dates, destination, preferences).
- Pay the full booking amount plus applicable convenience fee and GST on fees through the platform.
- Honour the cancellation policy agreed at the time of booking.
- Use the platform's chat and dispute channels for all communication with vendors.
- Not request or accept off-platform payments.

#### Vendor obligations

- Maintain accurate business details, KYC documents, fleet, driver, and package information.
- Place bids with genuine intent to fulfil the service at the quoted price.
- Maintain valid insurance and licenses required by law for the services offered.
- Respond to customer enquiries and negotiation within reasonable time.
- Deliver services as described in the accepted bid or package.
- Comply with Swabi's payout, wallet, and withdrawal rules.

#### Platform rights

- Swabi may suspend, restrict, or terminate accounts that violate these terms.
- Swabi may modify fees, plans, policies, or services with prior notice.
- Swabi acts as an intermediary and payment facilitator; it does not own or operate the underlying travel services.
- Disputes are resolved per Swabi's dispute resolution policy.

**Governing law:** Laws of India.

**Effect of breach:** Account suspension, forfeiture of pending payouts (where legally permitted), and legal action.

### 1.2 Privacy Policy

#### What we collect

- **Identity:** name, email, phone, government ID (vendors only).
- **Trip data:** destinations, dates, traveler details, preferences.
- **Payment data:** transaction reference, masked card details (via gateway only — Swabi never stores full card numbers).
- **Device data:** for tracking devices — location, battery, signal, events.
- **Usage data:** pages viewed, clicks, device type, IP.

#### How we use it

- To operate the marketplace: matching enquiries to vendors, processing bids and payments.
- To communicate: booking updates, alerts, disputes, marketing (opt-in).
- To improve: analytics, fraud detection, model training (aggregated only).
- To comply: GST, KYC, AML, law enforcement requests.

#### Who we share it with

- Vendors: only the information needed to serve your booking.
- Payment gateway: for payment processing.
- Tracking partner: for device operations.
- Authorities: when legally required.

#### Retention

- Financial records: 7 years.
- Location data (tracking): 90 days hot, 1 year cold.
- Account data: while active + 90 days after deletion request.

#### Your rights

- Access, correct, or delete your data.
- Withdraw marketing consent.
- Request data export.
- Raise a privacy grievance with Swabi support.

**Children:** Platform is for users 18 and above. Traveler details for minors are collected only as part of a family booking.

### 1.3 Community Guidelines

**Purpose:** Keep Swabi safe, respectful, and trustworthy for all users.

#### Prohibited behaviour

- Abusive, threatening, discriminatory, or obscene language.
- Harassment of customers, vendors, or Swabi staff.
- Sharing contact details to move off-platform.
- False information about trips, vendors, or identity.
- Posting bids without intent to fulfil.
- Manipulating prices, reviews, or bids (including fake accounts).
- Uploading malware, phishing links, or illegal content.
- Any activity violating Indian law.

#### Consequences

- First offence: warning + content removal.
- Repeat offence: temporary suspension.
- Serious offence: permanent ban + forfeiture of pending payouts.

**Reporting:** Use the "Report" button in chat or email support@swabi.io.

### 1.4 Customer and Vendor FAQs

#### Customer FAQs

- **Q: Is my payment safe?** A: Yes. It is held in escrow and released only against trip milestones.
- **Q: Can I cancel after paying?** A: Yes, per the cancellation policy. Refund percentage depends on how early you cancel.
- **Q: How do I get my money back if the vendor doesn't show up?** A: Raise a dispute from the booking page. Admin reviews within the SLA and issues a refund.
- **Q: Can I talk to the vendor before booking?** A: Yes, via chat. Do not share phone numbers.
- **Q: What if I need to change my dates?** A: Request changes via chat. The vendor must approve; both parties re-confirm.
- **Q: What is Swabi Plus?** A: A yearly membership (₹999) with waived convenience fee, priority support, free cancellation up to 7 days, and extra coins.

#### Vendor FAQs

- **Q: How do I become a verified vendor?** A: Register, upload KYC documents, pay the ₹1,999 verification fee, and wait for admin approval.
- **Q: How much does it cost to place a bid?** A: It depends on the bid amount and currency slab. Higher plans get a discount.
- **Q: When do I get paid?** A: Escrow funds are released to your wallet as trip milestones complete. You then request a withdrawal.
- **Q: What if a customer cancels?** A: You retain a percentage of the booking amount based on the cancellation tier.
- **Q: Can I offer a discount?** A: Yes, by lowering your bid or applying offers within the discount stacking cap.
- **Q: What if a tracking device is damaged?** A: Report via the device page. Damage deductions may apply to your deposit.

---

## SECTION 2 — BIDDING WORKFLOW

### 2.1 Enquiry and Bidding Process

#### Customer side

1. Customer posts an enquiry with destination, dates, travelers, budget, and preferences.
2. Enquiry status becomes live and the auction opens (default duration 24 hours).
3. Matched vendors are notified immediately (new vendors) or after 1 hour (existing vendors).
4. Customer reviews incoming bids, can chat, and negotiate.
5. Customer accepts a bid → booking created → payment flow starts.

#### Vendor side

1. Vendor sees the enquiry in "New Enquiries".
2. Vendor reviews requirements.
3. Vendor places a bid with amount, itinerary, inclusions, exclusions, terms, validity.
4. Bid cost is charged to the vendor's wallet before the bid is saved.
5. Vendor can chat, counter, or withdraw before acceptance.

#### Matching rules

- Vendors are matched by specialties AND service areas.
- If no match is found, fallback distributes to all eligible vendors.
- Fan-out is idempotent — vendors never receive duplicates.

#### Fairness rule (new vendors first)

- Vendors onboarded within the last N days receive the enquiry immediately.
- Existing vendors receive it after 1 hour.
- This gives new vendors a first-mover window.

### 2.2 Bid Management Rules

#### Placing a bid

- One active bid per vendor per enquiry.
- Bid amount must be > 0.
- Validity date cannot be in the past.

#### Editing a bid

- Vendors cannot edit a placed bid's amount directly.
- They must withdraw and re-bid, or negotiate.

#### Withdrawing a bid

- Allowed only while the bid is active.
- Wallet cost is refunded to the vendor's top-up bucket.
- Once withdrawn, the bid is no longer visible to the customer.

#### Accepting a bid

- Only the customer who posted the enquiry can accept.
- Only one bid per enquiry can be accepted.
- On acceptance, all other active bids are auto-rejected.
- Enquiry status moves to completed.

#### Bid expiry

- A bid expires on its `validUntil` date if set.
- Expired bids cannot be accepted.

#### Negotiation

- Customer and vendor can exchange offers via chat cards.
- Each new offer supersedes the previous pending offer.
- Both parties can accept, reject, or counter.
- On mutual agreement, the customer proceeds to booking.

### 2.3 Bid Fees and Refunds

#### How bid fees are calculated

1. The platform looks up the bid amount in the currency's slab table.
2. Slab defines: rate percentage, minimum charge, maximum charge.
3. Base cost = bid amount × rate %, clamped between min and max.
4. Plan discount is applied on the base cost.
5. Final cost = base cost − discount.

#### Plan discounts (current baseline)

- Basic: 0%
- Professional: 20%
- Enterprise: 40%

#### Plan discounts (proposed redesign)

- Free: 0%
- Silver: 10%
- Gold: 20%
- Platinum: 40%

#### When the fee is charged

- Immediately before the bid is saved.
- If the wallet has insufficient balance, the bid is rejected with a clear message.
- If the bid save fails, the fee is refunded automatically.

#### When the fee is refunded

- When the vendor withdraws the bid before acceptance.
- When the platform rejects the bid due to a system error.
- Never refunded after the customer accepts the bid.

#### Where the refund goes

- Back to the top-up balance in the vendor's wallet.
- Not to the withdrawable balance — top-up funds can only be spent on bids.

---

## SECTION 3 — PAYMENT AND ESCROW

### 3.1 Payment Workflow

#### Supported methods

- Razorpay (UPI, cards, net banking, wallets, EMI).
- Stripe (fallback, being activated).

#### Steps

1. Customer accepts a bid or books a package.
2. A booking is created with `paymentStatus = pending`.
3. Customer lands on the payment page. Fee breakdown is shown.
4. Backend creates a Razorpay order and returns an order ID.
5. Customer pays via the Razorpay checkout.
6. Razorpay sends a webhook to Swabi.
7. Swabi verifies the signature, marks the payment completed, and holds the escrow.
8. Booking status becomes upcoming, `paymentStatus = paid`.

#### Backup verify path

- If the webhook is delayed, the frontend's `/payments/verify` endpoint performs the same state transition.
- Both paths are idempotent — a payment can only be completed once.

#### Amount rules

- The amount is always sourced from the server (from `bid.amount` or `package.price`).
- Client-supplied amounts are ignored.

### 3.2 Escrow Rules

#### What escrow is

- The customer's payment is held by the platform and released to the vendor in stages.
- The vendor cannot spend escrow money until the admin releases it to their wallet.

#### Default milestone plan

- Milestone 1: `booking_confirmed` — 20% of the vendor payout.
- Milestone 2: `mid_trip` — 40%.
- Milestone 3: `trip_completed` — 40%.

#### How releases work

- Admin releases each milestone from the escrow page.
- The released amount is credited to the vendor's wallet.
- The vendor can then request a withdrawal.

#### Fully released

- When all milestones are released, the escrow status becomes released.

#### Auto-release

- Optional. If enabled, the `trip_completed` milestone releases automatically on booking completion.
- Default: disabled — admin must release manually.

Nothing is sent to the vendor's bank directly from escrow.

- All money moves escrow → wallet → withdrawal → bank.

### 3.3 Fee and Commission Structure

#### Customer pays

| Component | Formula | Notes |
|---|---|---|
| Base amount | Bid amount or package price | From server only |
| Discount | Capped at 25% of base | From offers |
| Discounted base | Base − discount | |
| Convenience fee | Discounted base × 1.5% | Waived for Swabi Plus |
| GST on fees | Convenience fee × 18% | Only on fees, not on base |
| Total payable | Discounted base + convenience fee + GST | Customer's charge |

#### Vendor receives

| Component | Formula |
|---|---|
| Commission | Base amount × 8% (locked at escrow creation) |
| Vendor payout | Base amount − commission |
| Payout on completion | Released in milestones |

#### Platform earns

| Component | Source |
|---|---|
| Commission | From each booking |
| Convenience fee | From each customer |
| GST on fees | Remitted to government |
| Subscriptions | Vendor plans |
| Bid costs | Every vendor bid |
| Membership | Swabi Plus |
| Verification fee | Vendor onboarding |

#### Discount rules

- Multiple discounts cannot stack beyond 25% of base.
- Coins redemption also caps at 25% of total payable.

### 3.4 Payment Failure Handling

#### If the payment fails

- Razorpay sends a `payment.failed` webhook.
- Payment status becomes failed.
- Booking stays pending.
- Customer can retry from the payment page.

#### If the payment stays pending too long

- The customer can reopen the payment page.
- The same Razorpay order is reused if still valid.
- Otherwise a new order is created.

#### If the customer's money was debited but the booking is not paid

- Within 24 hours, Razorpay auto-refunds or auto-reconciles.
- If unresolved, customer raises a dispute.
- Admin reconciles with Razorpay logs.

#### If the webhook never arrives

- The `/payments/verify` endpoint performs the state transition.
- If both fail, admin manually reconciles using the Razorpay payment ID.

---

## SECTION 4 — CANCELLATION AND REFUNDS

### 4.1 Cancellation Policy

Refund to customer and retention by vendor depends on how early the cancellation is made.

| Time before trip | Customer refund | Vendor retains |
|---|---:|---:|
| More than 30 days | 100% | 0% |
| 15 to 30 days | 75% | 25% |
| 7 to 14 days | 50% | 50% |
| Less than 7 days | 25% | 75% |
| No-show | 0% | 100% |

#### Special rules

- Commission on cancelled bookings is refunded to the customer — the platform absorbs it.
- The gateway fee is absorbed by the platform.
- Swabi Plus members get free cancellation up to 7 days (100% refund).

### 4.2 Refund Process

#### Trigger points

- Customer cancels a booking.
- Admin resolves a dispute in the customer's favour.
- Platform cancels a booking for operational reasons.
- Partial refund on a dispute resolution split.

#### How refunds are processed

1. Refund request is validated against the policy tier.
2. Amount is calculated (customer refund % of base).
3. Razorpay refund API is called with the original payment ID.
4. Refund row is created in the refunds table.
5. Payment status becomes refunded or partially_refunded.
6. Booking status becomes cancelled.

#### Timelines

- Initiated within 24 hours of approval.
- Credited to the customer's original payment method in 5–7 business days.
- Coins clawback (if awarded at booking) applied on the refund.

#### Method

- Original source only. Card → card, UPI → UPI.
- Alternative methods only if the original is unavailable, with customer consent.

### 4.3 Exceptions

#### No-show

- Vendor must mark `no_show` with reason.
- Customer gets 0% refund.
- Customer can dispute — evidence required.

#### Emergency cancellation

- Documented medical or family emergency: admin can override the tier.
- Requires proof (medical certificate, etc.).
- Case-by-case decision.

#### Unexpected disruption

- Natural disaster, political unrest, or government ban at destination.
- Platform-initiated cancellation: full refund.
- No commission charged to the vendor.

#### Platform-related issues

- If Swabi's systems cause a payment or booking failure, the customer gets a full refund.
- Platform absorbs all gateway costs.

---

## SECTION 5 — WALLET AND PAYOUTS

### 5.1 Wallet Rules

#### Two buckets for vendors

- **Escrow balance:** credited from released milestones. Withdrawable.
- **Top-up balance:** funded by the vendor. Not withdrawable, used for bid costs.
- **Locked balance:** amount reserved during a pending withdrawal.

#### Customer wallet

- Only holds Swabi Coins (loyalty).
- No withdrawable money.

#### Ledger

- Every credit and debit is logged in the wallet ledger.
- Each entry shows: type, amount, running balance, reference, timestamp.

#### Ledger types

- `escrow_release`, `topup`, `bid_cost`, `bid_cost_refund`
- `withdrawal_lock`, `withdrawal_completed`, `withdrawal_rejected`
- `credit_consume`, `credit_refund`, `credit_grant`, `credit_purchase`
- `coin_earn`, `coin_redeem`, `admin_adjust`

### 5.2 Top-Up and Withdrawal Rules

#### Top-up (vendor adds money)

- Minimum: ₹100
- Maximum: ₹1,00,000 per transaction
- Fee: none
- Method: Razorpay
- Goes to `topupBalance` (not withdrawable)
- Purpose: fund bid costs

#### Withdrawal (vendor requests money)

- Minimum: ₹500
- Maximum: the entire withdrawable balance
- Eligibility: only from `escrowBalance`
- Payout methods: Razorpay Route or manual bank transfer
- Bank details must be on file

#### Restrictions

- `topupBalance` cannot be withdrawn.
- `lockedBalance` reduces the withdrawable amount until the request is resolved.
- Multiple pending withdrawals allowed but each locks its own amount.

### 5.3 Vendor Payout Process

#### Steps

1. Vendor requests withdrawal with an amount.
2. Funds are locked in `lockedBalance`.
3. Request enters pending state, visible to admin.
4. Admin reviews and approves or rejects.
5. On approval:
   - If Razorpay Route account exists → automatic transfer.
   - Otherwise → status moves to approved awaiting manual NEFT.
6. Admin marks paid with a UTR number.
7. On completion:
   - `lockedBalance` decreases.
   - `escrowBalance` decreases.
   - `lifetimeWithdrawn` increases.
8. On rejection:
   - `lockedBalance` unlocks back to `escrowBalance`.
   - Reason is shown to the vendor.

#### Timelines

- Approval: target 4 business hours.
- Automatic Route transfer: minutes.
- Manual NEFT: 1 business day.

**Fees:** Swabi does not charge a fee on withdrawals.

---

## SECTION 6 — DISPUTE RESOLUTION

### 6.1 Dispute Process

#### Who can raise

- Customer, for any booking-related issue.
- Vendor, for payment or customer-behaviour issues.

#### When to raise

- Within 7 days of booking completion.
- Or immediately if the vendor fails to show.

#### How to raise

From the booking detail page → "Raise dispute". Enter reason, description, and evidence (images, screenshots, receipts).

#### Reasons

- Vendor no-show
- Service not as described
- Unsafe or unprofessional conduct
- Damage or loss of property
- Payment issue
- Other

#### What happens next

- Dispute status becomes open.
- Both parties are notified.
- Admin sees the dispute in the queue.

### 6.2 Investigation and Resolution

#### Investigation

1. Admin reviews the evidence from both parties.
2. Admin may request additional documents.
3. Admin may contact both parties via platform chat or phone.
4. Timeline: within 7 business days.

#### Possible outcomes

- Full refund to customer.
- Partial refund (split).
- Full release to vendor.
- Additional actions (warning, suspension).

#### Resolution

- Admin writes a resolution note.
- Dispute status becomes `resolved_customer`, `resolved_vendor`, or `resolved_split`.
- Refund and payout actions are triggered automatically if applicable.
- Both parties are notified.

#### Escalation path

- Level 1: Support agent
- Level 2: Trust & Safety lead
- Level 3: Admin (for > ₹50,000 disputes or legal threats)
- Legal: for court notices or regulatory complaints

### 6.3 Escalation Procedures

#### When to escalate

- Dispute value exceeds ₹50,000.
- Dispute involves safety, injury, or police complaint.
- Either party threatens legal action.
- Repeated disputes by the same user.
- Media or regulatory interest.

#### Expected resolution timelines

| Severity | First response | Resolution |
|---|---|---|
| Low (payment query) | 4 hours | 2 business days |
| Medium (service issue) | 8 hours | 5 business days |
| High (no-show, damage) | 4 hours | 7 business days |
| Critical (safety, legal) | 1 hour | 24 hours (interim), then per case |

#### Contact points

- Support chat in-app
- Email: support@swabi.io
- Emergency: emergency@swabi.io (SOS only)

---

## SECTION 7 — VENDOR VERIFICATION

### 7.1 Vendor Onboarding

#### Registration steps

1. Create account with email and phone.
2. Enter business details: company name, registration number, tax ID, address, contact person.
3. Upload KYC documents:
   - Business license or registration certificate
   - GST certificate (if applicable)
   - PAN card
   - Identity proof (Aadhaar, Passport, or Driving License)
   - Address proof (utility bill or rent agreement)
   - Cancelled cheque or bank statement (for payouts)
4. Declare service areas and specialties.
5. Pay the one-time verification fee of ₹1,999.
6. Submit for admin review.

#### After submission

- Status: pending review.
- Admin reviews within 3 business days.
- On approval, the vendor badge appears on all listings.

### 7.2 Verification Rules

#### What admin checks

- Documents are valid and match business details.
- Business is legally registered.
- Bank account details match the business name (or an authorised signatory).
- No prior ban or fraud history on the platform.

#### Possible outcomes

- Approved → vendor is verified, badge enabled.
- Rejected → clear reason given, vendor can reapply after 30 days.
- Additional documents requested → vendor has 7 days to respond.

#### Verification expiry

- Verifications are valid for 1 year, then require renewal.
- Expired verification prevents new listings or bids until renewed.

### 7.3 Account Restrictions

#### Grounds for suspension

- Providing fake documents or false information.
- Multiple confirmed disputes in the customer's favour.
- Non-compliance with refund decisions.
- Abusive behaviour with customers or staff.
- Repeated no-shows.
- Attempting to move customers off-platform.
- Fraudulent bids, payments, or reviews.

#### Stages of restriction

1. Warning
2. Temporary suspension (7 to 30 days)
3. Permanent ban + forfeiture of pending payouts (where legally permitted)

#### Appeal

- Vendors can appeal within 14 days.
- Admin reviews and issues a final decision.

### 7.4 Vendor Plans

#### Current baseline

| Plan | Monthly | Vehicles | Drivers | Activities | Packages | Bid Credits |
|---|---:|---:|---:|---:|---:|---:|
| Basic | ₹499 | 10 | 5 | 5 | 5 | 20 |
| Professional | ₹999 | 50 | 25 | 10 | 20 | 100 |
| Enterprise | ₹2,999 | Unlimited | Unlimited | Unlimited | Unlimited | Unlimited |

#### Proposed redesign

| Plan | Listings | Bid discount | Overage | Devices included |
|---|---:|---:|---|---|
| Free | TBD | 0% | — | 0 |
| Silver | 10 | 10% | — | 5 |
| Gold | 20 | 20% | $1/listing | 15 |
| Platinum | Unlimited | 40% | — | Unlimited |

#### What plans include

- Higher listing caps
- Better bid discounts
- Priority support (Gold and Platinum)
- More devices for tracking (Silver and above)
- Custom branding (Gold and Platinum)
- Dedicated account manager (Platinum)

#### Billing

- Monthly or yearly cycle.
- Auto-renew optional.
- Upgrade: immediate. Downgrade: next cycle.

---

## SECTION 8 — CUSTOMER, VENDOR, AND ADMIN WORKFLOWS

### 8.1 Customer Workflow

1. Sign up with email and phone.
2. Complete profile with name, contact, preferences.
3. Post an enquiry: destination, dates, travelers, budget, preferences.
4. Auction runs for 24 hours. Vendors submit bids.
5. Review bids: compare prices, itineraries, inclusions.
6. Chat and negotiate with shortlisted vendors.
7. Accept a bid → booking created.
8. Pay securely via Razorpay. Money held in escrow.
9. Track trip: chat with the vendor, track via device.
10. Complete trip. Rate the vendor (post-trip).
11. Raise dispute if something goes wrong within 7 days.

### 8.2 Vendor Workflow

1. Register and complete business details.
2. Upload KYC documents.
3. Pay verification fee.
4. Get verified by admin.
5. Subscribe to a plan (optional, for higher caps and discounts).
6. Set up fleet, drivers, activities, packages.
7. Browse live enquiries in "New Enquiries".
8. Place bids (wallet charged).
9. Chat and negotiate with customers.
10. Win a booking → trip assigned.
11. Assign drivers and vehicles.
12. Assign tracking devices to the trip.
13. Complete the trip → booking completed.
14. Get paid in milestones released from escrow to wallet.
15. Request a withdrawal → admin approves → bank transfer.

### 8.3 Booking and Trip Lifecycle

| Stage | Trigger | What happens |
|---|---|---|
| Enquiry posted | Customer action | Fan-out to vendors |
| Bid placed | Vendor action | Wallet charged |
| Bid accepted | Customer action | Booking created (pending) |
| Payment made | Customer action | Escrow created |
| Booking confirmed | Payment success | Vendor notified |
| Trip started | Day of trip | Device tracking active |
| Trip mid-point | Vendor or admin | Mid-trip milestone released |
| Trip completed | Vendor action | Final milestone released |
| Escrow closed | All milestones released | Vendor can withdraw |
| Withdrawal approved | Admin action | Money moves to vendor bank |
| Review submitted | Customer action | Feedback recorded |

### 8.4 Admin and Support Procedures

#### Daily responsibilities

- Review and approve vendor verification requests.
- Approve or reject withdrawal requests.
- Release escrow milestones as trips progress.
- Monitor payment failures and retries.
- Review open disputes.
- Check tracking device alerts (SOS, geofence, offline).

#### Escrow release procedure

1. Open `/admin/escrows`.
2. Review the escrow's booking status.
3. Verify trip progress externally if needed.
4. Click "Release next milestone".
5. Money credited to vendor wallet.
6. Audit log entry created.

#### Withdrawal approval procedure

1. Open `/admin/withdrawals`.
2. Filter by "Pending".
3. Reveal bank details (audit-logged).
4. Verify the amount and vendor identity.
5. Approve (Route auto-transfer) or reject with reason.
6. If manual, mark paid with UTR after NEFT.

#### Dispute handling procedure

1. Review evidence from both parties.
2. Request additional documents if needed.
3. Investigate via chat, call, or third-party records.
4. Resolve within SLA with a written note.
5. Trigger refund or release based on outcome.

#### Exceptional cases

- Payment disputed with bank → cooperate with Razorpay, provide evidence.
- Fraud detected → immediate suspension + report.
- SOS alert → escalate to Trust & Safety within SLA.
- Legal notice → route to legal team immediately.

---

## SECTION 9 — TRAVEL KNOWLEDGE

### 9.1 Destination Guides

#### Kashmir (India)

- **Best time:** March to October. April and October are ideal.
- **Peak:** June (school holidays) and September (autumn).
- **Major attractions:** Dal Lake, Gulmarg, Pahalgam, Sonmarg, Mughal Gardens, Shikara rides, houseboats.
- **Travel requirements:** valid ID for houseboats; Inner Line Permits for certain border areas.
- **Weather notes:** cold nights even in summer; heavy snowfall Dec–Feb.

#### Maldives

- **Best time:** November to April (dry season).
- **Peak:** December to March.
- **Major attractions:** Malé, overwater villas, snorkeling, diving, sandbanks, dolphin cruises.
- **Travel requirements:** valid passport (6 months validity), return ticket.
- **Weather notes:** monsoon May–October, ideal for budget travel.

#### Bali (Indonesia)

- **Best time:** April to October.
- **Peak:** July and August.
- **Major attractions:** Ubud, Uluwatu, Seminyak, Tanah Lot, rice terraces, temples.
- **Travel requirements:** valid passport, visa on arrival for most nationalities.
- **Local customs:** dress modestly at temples, sarongs required.

#### Goa (India)

- **Best time:** November to February.
- **Peak:** December and January.
- **Major attractions:** North Goa beaches, South Goa quiet escapes, Old Goa churches, spice plantations.
- **Travel requirements:** no special permits for Indian citizens.
- **Weather notes:** monsoon June–September with rough seas.

#### Rajasthan (India)

- **Best time:** October to March.
- **Peak:** December and January.
- **Major attractions:** Jaipur, Udaipur, Jodhpur, Jaisalmer, Mount Abu, Ranthambore.
- **Travel requirements:** no special permits.
- **Weather notes:** extremely hot in summer (April–June).

### 9.2 Activity Descriptions

#### Trekking

- **Duration:** half-day to multi-day.
- **Difficulty:** easy (day hikes) to strenuous (Himalayan treks).
- **Suitability:** healthy adults; not recommended for children under 8 or seniors with heart conditions.
- **Safety:** guide mandatory, hydration essential, acclimatisation for altitude.

#### Scuba Diving

- **Duration:** 2 hours to multi-day courses.
- **Difficulty:** beginner to advanced.
- **Suitability:** age 10+, no asthma or heart conditions.
- **Safety:** certified instructor required; medical declaration.

#### White Water Rafting

- **Duration:** 2 to 4 hours.
- **Difficulty:** grade I to V.
- **Suitability:** age 12+ (varies by river grade).
- **Safety:** helmet, life vest, trained guide.

#### Paragliding

- **Duration:** 10 to 30 minutes.
- **Difficulty:** tandem-friendly.
- **Suitability:** age 8+, weight limit usually 100kg.
- **Safety:** certified operator, weather-dependent.

#### Wildlife Safari

- **Duration:** 2 to 4 hours per session.
- **Difficulty:** easy.
- **Suitability:** all ages; suitable for families.
- **Safety:** stay in vehicle, follow guide rules, silent zones.

#### Cultural Walks

- **Duration:** 2 to 3 hours.
- **Difficulty:** easy.
- **Suitability:** all ages.
- **Notes:** dress modestly at religious sites.

#### Houseboat Stay

- **Duration:** overnight to multi-day.
- **Difficulty:** easy.
- **Suitability:** couples, families, seniors.
- **Notes:** verify safety equipment; choose registered operators.

#### Camping

- **Duration:** overnight to multi-day.
- **Difficulty:** easy to moderate.
- **Suitability:** families and groups.
- **Notes:** check weather, altitude, and amenities.

### 9.3 Travel Tips and Guidelines

#### Packing advice

- Layer clothing for variable weather.
- Carry sunscreen, insect repellent, and basic medicines.
- Bring power bank and universal adapter.
- For adventure trips: sturdy shoes, quick-dry clothes, rain jacket.
- For cultural sites: modest clothing, scarf for head covering.

#### Local customs

- Remove shoes before entering temples and homes in India.
- Dress modestly at religious sites.
- Ask before photographing people.
- Carry small cash for tips and local purchases.
- Respect local prayer times and festivals.

#### Seasonal conditions

- Himalayan regions: cold nights year-round; snow Dec–Feb.
- Coastal areas: monsoon June–September with rough seas.
- Desert regions: extreme heat April–June; cold nights Dec–Jan.
- Tropical destinations: brief afternoon showers year-round.

#### Travel safety

- Share your itinerary with family (Swabi's family tracking link helps).
- Carry a government ID at all times.
- Keep emergency numbers saved: police 100, ambulance 108, tourist helpline 1363.
- Avoid isolated areas at night.
- Follow the tracking device's SOS instructions if in danger.
- Stay in touch via platform chat for anything booking-related.

#### Before you travel

- Confirm the booking 48 hours before departure.
- Reconfirm pickup time with the vendor.
- Check weather forecast for the destination.
- Verify device assignment (if opted).
- Download offline maps and translation tools.

#### During the trip

- Carry booking confirmation and ID.
- Use platform chat for booking changes.
- Report any issue to the vendor first; escalate to support if unresolved.
- Use SOS on the tracking device in emergencies.
- Rate your vendor after the trip.

#### After the trip

- Submit any feedback or complaint within 7 days.
- Rate vendors honestly.
- Leave the tracking device with the vendor as agreed.
- Claim coins and use them on your next booking.
