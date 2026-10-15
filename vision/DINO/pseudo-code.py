# initialize
student = ViT()
teacher = copy(student)

# teacher does not receive gradients
stop_gradient(teacher)

center = zeros()

for images in dataloader:
    # 1. multi-crop augmentation
    global_1 = global_crop(images)
    global_2 = global_crop(images)

    local_1 = local_crops(images)
    local_2 = local_crops(images)
    local_3 = local_crops(images)

    teacher_views = [
        global_1,
        global_2
    ]

    student_views = [
        global_1,
        global_2,
        local1,
        local2,
        local3
    ]

    # 2. teacher forward
    teacher_outputs = []

    for view in teacher_views:
        cls = teacher.vit(view).CLS

        logits = teacher.head(cls)

        # centering
        logits = logits - center

        # sharpening
        prob = softmax(
            logits / teacher_temperature
        )

        teacher_outputs.append(
            stop_gradient(prob)
        )
    
    # 3. student forward
    student_outputs = []

    for view in student_views:

        cls = student.vit(view).CLS

        logits = student.head(cls)

        prob = softmax(
            logits / student_temperature
        )

        student_outputs.append(prob)

    # 4. cross-view loss
    loss = 0

    for teacher_view in teacher_outputs:
        for student_view in student_outputs:

            if same_view(
                teacher_view,
                student_view
            ):
                continue
                
            loss += cross_entropy(
                teacher_view,
                student_view
            )
    
    # 5. update student
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # 6. EMA update teacher
    for param_t, param_s in zip(
        teacher.parameters(),
        student.parameters()
    ):
        param_t = (
            momentum * param_t
            +
            (1 - momentum) * param_s
        )
    
    # 7. update center
    batch_center = mean(teacher_logits)

    center = (
        center_momentum * center
        +
        (1 - center_momentum) * batch_center
    )