<template>
    <div class="page">
        <div class="menu">
            <div class="item">
                <div class="title">model</div>
                <el-select class="value" size="small" v-model="params.model" @change="update">
                    <el-option v-for="model in models" :key="model" :value='model'/>
                </el-select>
                </div>
            <div class="item">
                <div class="title">input</div>
                <el-select class="value" size="small" v-model="params.input" @change="update">
                    <el-option value='static/images/cat_dog.png'/>
                    <el-option value='static/images/spider.png'/>
                    <el-option value='static/images/snake.jpg'/>
                </el-select>
            </div>
            <div class="item"><div class="title">target</div><el-input class="value" type='number' size="small" v-model="params.target"  @change="update"/></div>
        </div>
        <div class="network">
            <div class="input"><img :src="params.input" crossorigin='anonymous'/></div>
            <div class="sliency">
                <div class="image-wrapper"><img class="image" :src="res.colorful" crossorigin='anonymous'/><div class="caption">Gradient</div></div>
                <div class="image-wrapper"><img class="image" :src="res.grayscale" crossorigin='anonymous'/><div class="caption">Saliency</div></div>
                <div class="image-wrapper"><img class="image" :src="res.grad_x_image" crossorigin='anonymous'/><div class="caption">Saliency * Image</div></div>
            </div>
            <div class="sliency">
                <div class="image-wrapper"><img class="image" :src="res.guided_colorful" crossorigin='anonymous'/><div class="caption">Guided Gradient</div></div>
                <div class="image-wrapper"><img class="image" :src="res.guided_grayscale" crossorigin='anonymous'/><div class="caption">Guided Saliency</div></div>
                <div class="image-wrapper"><img class="image" :src="res.guided_grad_x_image" crossorigin='anonymous'/><div class="caption">Guided Saliency * Image</div></div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    name: 'saliency',
    data() {
        return {
            models: [],
            activations: [],
            params: {
                model: 'alexnet',
                input: 'static/images/cat_dog.png',
                target: 243
            },
            res: {}
        };
    },
    created() {
        this.config()
        this.update()
    },
    sockets: {
        response_saliecy(data) {
            this.res = data
        },

        models(data) {
            this.models = data
        }
    },
    methods: {
        config() {
            this.$socket.emit('get_models')
        },
        update() {
            this.$socket.emit("saliency", this.params);
        }
    }
};
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
.page {
    .network {
        display: flex;
        flex-flow: row;

        align-items: center;
        justify-items: center;

        .input {
            flex: 0 0 auto;
        }

        .sliency {
            flex: 1 1 auto;
            display: flex;
            flex-flow: column;
            align-items: center;
            justify-items: center;
            img {
                padding: 10px;
            }
        }
    }
}

</style>
