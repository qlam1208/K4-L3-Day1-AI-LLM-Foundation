# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi tăng temperature từ 0.0 lên 1.5, phản hồi chuyển từ tính tất định, chọn lọc các sự thật phổ biến/số liệu thống kê an toàn (như xuất khẩu cà phê/hạt điều ở mức 0.0–0.5) sang nội dung mang tính văn hóa độc đáo và ít phổ biến hơn (như con Mèo trong 12 con giáp ở mức 1.5). Tuy nhiên, ở mức 1.5 tính ngẫu hứng quá cao khiến văn phong dễ bị rời rạc, xuất hiện ký tự lạ và giảm độ ổn định so với các mức thấp.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Em sẽ đặt temperature khoảng 0.0 đến 0.2 (hoặc 0.1). Vì chatbot chăm sóc khách hàng yêu cầu độ chính xác cao và tính nhất quán tuyệt đối về thông tin (chính sách, giá cả, giải quyết khiếu nại), việc đặt temperature thấp sẽ giảm thiểu tối đa rủi ro ảo giác hay bịa đặt dữ liệu, đồng thời đảm bảo câu trả lời luôn chuẩn mực và đáng tin cậy.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> GPT-4o đắt hơn GPT-4o-mini khoảng 16.7 lần cho workload này ($105/ngày so với $6.3/ngày cho 10.5 triệu token output).
Nên dùng GPT-4o khi: Tác vụ đòi hỏi suy luận phức tạp, đa bước và yêu cầu độ chính xác tuyệt đối như phân tích hợp đồng pháp lý, kiểm toán tài chính hoặc viết code giải thuật khó.
Nên dùng mini khi: Các tác vụ lặp lại quy mô lớn nhưng cấu trúc đơn giản như phân loại ý định câu hỏi (intent classification), chatbot tra cứu FAQ thường gặp hoặc tóm tắt đoạn văn ngắn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi khác biệt rõ rệt về phong cách tiếp cận: persona "giáo viên tiểu học" giải thích ngắn gọn, từ ngữ đơn giản và sử dụng ẩn dụ đời thường trực quan (như cuốn sổ viết bằng mực không thể xóa), trong khi persona "chuyên gia tài chính" dùng thuật ngữ chuyên môn sâu (sổ cái phân tán, mật mã học, tính bất biến) và cấu trúc phân tích trang trọng. System prompt đóng vai trò như "chỉ thị đạo diễn", định hình persona, tông giọng (tone) và đối tượng tiếp nhận mục tiêu cho toàn bộ câu trả lời. Nhờ đó, cùng một câu hỏi nhưng mô hình có thể linh hoạt chuyển đổi hoàn toàn kho từ vựng và mức độ trừu tượng để phục vụ đúng ngữ cảnh mà người dùng mong muốn.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn tiếng Việt ~110 từ, công thức ước lượng số từ / 0.75 cho kết quả ~150 token, trong khi tiktoken đếm được 145 token trên GPT-4o (chênh ~4%) nhưng lên tới 256 token trên GPT-4 cũ (chênh lệch tới ~70%). Tiếng Việt tốn nhiều token hơn tiếng Anh vì tokenizer (BPE) được tối ưu chủ yếu cho ngữ liệu tiếng Anh; các ký tự có dấu thanh và dấu phụ trong tiếng Việt chiếm nhiều byte UTF-8 và thường bị thuật toán phân rã thành nhiều mảnh subword nhỏ thay vì giữ nguyên 1 token cho cả từ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng tương tác trực tiếp với người dùng (như chatbot, trợ lý ảo, sinh nội dung dài), nơi việc tối ưu thời gian nhận token đầu tiên (TTFT) giúp mang lại trải nghiệm mượt mà, người dùng có thể đọc phản hồi ngay lập tức thay vì phải chờ đợi nhiều giây trước màn hình loading. Ngược lại, non-streaming lại phù hợp hơn cho các tác vụ xử lý ngầm (backend batch jobs, cron jobs), các hệ thống tự động hóa machine-to-machine, hoặc khi đầu ra yêu cầu định dạng có cấu trúc (như JSON) cần được nhận trọn vẹn để kiểm tra tính hợp lệ (schema validation) trước khi xử lý logic tiếp theo.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp tăng dần thời gian chờ sau mỗi lần thất bại (1s, 2s, 4s, 8s...), tạo cho server một "khoảng thở" ngày càng rộng để giải phóng tài nguyên và tự phục hồi khi quá tải. Nếu hàng nghìn client cùng retry với delay cố định (ví dụ đúng 1 giây), hệ thống sẽ gặp hiện tượng "Thundering Herd" (Retry Storm): toàn bộ client sẽ đồng loạt dội một đợt sóng request mới vào server ở cùng một thời điểm, tiếp tục làm nghẽn hệ thống và gây ra chuỗi sập đổ dây chuyền (cascading failure) khiến server không thể hồi phục.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Em chọn persona: Trợ giảng môn AI thân thiện.
System prompt: "Bạn là trợ giảng thân thiện của khóa học AI, giải thích các khái niệm kỹ thuật dễ hiểu và luôn trả lời ngắn gọn bằng tiếng Việt."
Giải thích lựa chọn từ ngữ:
"ngắn gọn": Giúp hạn chế tối đa output token để tiết kiệm chi phí API và giảm độ trễ (latency), đồng thời giữ cho giao diện dòng lệnh (CLI) gọn gàng, tránh làm tràn màn hình terminal.
"bằng tiếng Việt": Đảm bảo mô hình luôn đồng nhất ngôn ngữ phản hồi ngay cả khi người dùng hỏi các câu có chứa thuật ngữ kỹ thuật hoặc đoạn code tiếng Anh.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> > Hạn chế lớn nhất: Không có bộ nhớ dài hạn (non-persistent memory); trợ lý quên hoàn toàn nội dung và ngữ cảnh của các phiên chat trước đó ngay sau khi kết thúc, dẫn đến trải nghiệm rời rạc và yêu cầu người dùng phải nhắc lại thông tin cũ nếu muốn tiếp tục thảo luận sâu.
Cải thiện đề xuất: Triển khai bộ nhớ bất biến (in-memory persistence) bằng cách lưu trữ lịch sử chat vào một cấu trúc dữ liệu (ví dụ: dictionary hoặc list) trong bộ nhớ RAM của chương trình; mỗi lần người dùng gửi tin nhắn, hệ thống sẽ tải toàn bộ lịch sử cũ, thêm vào context (context stuffing) và truyền toàn bộ cho API. Cách này đơn giản, không cần cơ sở dữ liệu, nhưng chỉ duy trì bộ nhớ trong thời gian chạy (cho đến khi tắt ứng dụng).


---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
